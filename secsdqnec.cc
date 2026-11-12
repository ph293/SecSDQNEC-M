// =====================================================================================
//  SecSDQNEC-M : Secure middleware-enabled SDN + DDQN task offloading  (ns-3.40 + OFSwitch13)
//
//  Topology (Fig. 1 of the paper)
//    IoT-UAV layer      : N UAVs, 802.11ac Wi-Fi, RandomWalk2d mobility at 50 m altitude
//    Edge middleware    : Wi-Fi AP + application-level middleware (classifier, local detector,
//                         trust engine, local compute 4 GHz) connected to OpenFlow switch S1
//    Data plane         : OpenFlow 1.3 switches S1 -> {S2 | S3} -> S4  (two alternative paths)
//    Control plane      : SecController (OFSwitch13Controller) = SDN controller + DDQN agent +
//                         global detector + response module; installs a flow rule per task
//    Compute layer      : Edge-1, Edge-2 (10 GHz), Cloud (40 GHz, 100 Mbps / 40 ms WAN link)
//    Attackers          : DDoS flooding host on S1, compromised edge, fake-task UAVs
//
//  Modes:  --mode=collect   generate labelled detector windows (CSV)
//          --mode=traindet  train/evaluate the autoencoder detectors from the CSVs
//          --mode=train     train the DQN/DDQN agent of one method inside ns-3
//          --mode=eval      evaluate a trained method, append one row to results/eval.csv
// =====================================================================================
#include "nn.h"

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/ofswitch13-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/wifi-module.h"

#include <chrono>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>

using namespace ns3;
using secnn::Vec;

// ------------------------------------------------------------------ configuration
struct Cfg
{
    std::string mode = "eval", method = "Proposed", scenario = "No attack";
    std::string out = "results", models = "models";
    uint32_t nUav = 20, seed = 1, windows = 300;
    double lambda = 0.5, window = 15.0, simTime = 40.0, warmup = 2.0;
    double ddosMbps = 92.0, dropProb = 0.6; // attack intensity (sensitivity study)
};
static Cfg g;

struct Method
{
    bool ddqn, mw, paths, trust, secReward, response, detect;
    bool staticIds = false; // conventional rule-based IDS (z-score threshold) instead of the AE
    bool mask = true;       // trust-aware action masking (ablation: Proposed_wo_mask)
};
static Method M;

static Method
GetMethod(const std::string& n)
{
    //          ddqn  mw     paths  trust  secRew response detect
    if (n == "DQNEC")
        return {false, false, false, false, false, false, false};
    if (n == "DDQNEC")
        return {true, false, false, false, false, false, false};
    if (n == "SDQNEC")
        return {true, false, true, false, false, false, false};
    if (n == "Proposed")
        return {true, true, true, true, true, true, true};
    if (n == "SDQNEC_IDS") // SDQNEC + conventional IDS at the controller (threshold + blocking)
        return {true, false, true, false, false, true, true, true};
    if (n == "Proposed_wo_mask") // trust in state + security reward, but no action masking
        return {true, true, true, true, true, true, true, false, false};
    if (n == "Proposed_wo_secreward") // masking kept, security term removed from the reward
        return {true, true, true, true, false, true, true};
    if (n == "Proposed_wo_trust")
        return {true, true, true, false, false, true, true};
    if (n == "Proposed_wo_middleware")
        return {true, false, true, true, true, true, true};
    if (n == "Proposed_wo_security")
        return {true, true, true, false, false, false, false};
    if (n == "Greedy" || n == "Random") // non-learning references (same actions as SDQNEC)
        return {true, false, true, false, false, false, false};
    if (n == "Heuristic")
        return {true, true, true, false, false, false, true}; // used only for --mode=collect
    NS_FATAL_ERROR("unknown method " << n);
}

// sites: 0 local UAV, 1 middleware, 2 edge-1, 3 edge-2, 4 cloud
static const double CPU_HZ[5] = {1e9, 4e9, 10e9, 10e9, 40e9};
static const double COST_GC[5] = {0.0, 0.010, 0.030, 0.030, 0.020};
static const double KAPPA_E[5] = {1e-9, 4.8e-10, 3e-10, 3e-10, 1.6e-10}; // J per cycle
static const double MAXQ[5] = {2.0, 3.0, 4.0, 4.0, 20.0};
static const double RADIO_W = 0.8;    // UAV Wi-Fi radio power while transmitting
static const double MW_PROC = 0.003;  // middleware screening/classification delay
static const double RESULT_RATIO = 0.05;
static const double PATH_CAP = 100e6; // switch path capacity (bit/s)

static std::vector<std::pair<int, int>> g_actions; // (site, path) ; path -1 = default hash path

static void
BuildActions()
{
    g_actions.clear();
    if (M.mw)
    {
        g_actions.push_back({0, -1});
        g_actions.push_back({1, -1});
    }
    for (int s = 2; s <= 4; ++s)
    {
        if (M.paths)
        {
            g_actions.push_back({s, 0});
            g_actions.push_back({s, 1});
        }
        else
        {
            g_actions.push_back({s, -1});
        }
    }
}

// ------------------------------------------------------------------ global state
static Ptr<UniformRandomVariable> U;
static std::unique_ptr<secnn::DqnAgent> g_agent;
static secnn::Autoencoder g_aePath, g_aeUav, g_aeSrv;
static bool g_learn = false, g_greedy = false;
static std::string g_policy = "agent";
static std::map<std::string, int> g_reason;
static bool g_dbg = getenv("SECDBG") != nullptr;
#define DBG(x) if (g_dbg) std::cerr << Simulator::Now().GetSeconds() << " " << x << std::endl

// attack state
static bool g_ddos = false, g_fake = false;
static int g_ddosPath = 0, g_compSite = -1;
static double g_attackStart = 1e9;
static std::string g_curScenario = "No attack";
static int g_windowIdx = 0;

// trust
static double g_trustSite[5] = {1, 1, 1, 1, 1};
static double g_trustPath[2] = {1, 1};
static std::vector<double> g_trustUav;
static bool g_ddosMitigated = false;

// v3: beta-reputation trust for servers (Josang & Ismail). r = positive evidence (tasks
// answered), b = negative evidence (silent drops, timeouts, detector alarms).
// trust = (r + 1) / (r + b + 2); evidence fades with factor TRUST_FADE per second towards the prior.
static const double TRUST_PRIOR_R = 4.0; // prior belief: servers start trusted (trust 0.83)
static const double TRUST_FADE = 0.95;
static double TRUST_MASK = 0.4;          // sites below this trust are masked (--maskTh for sensitivity)
static double g_rPos[5] = {TRUST_PRIOR_R, TRUST_PRIOR_R, TRUST_PRIOR_R, TRUST_PRIOR_R, TRUST_PRIOR_R};
static double g_rNeg[5] = {0, 0, 0, 0, 0};

static void
SiteEvidence(int s, double pos, double neg)
{
    g_rPos[s] += pos;
    g_rNeg[s] += neg;
    g_trustSite[s] = (g_rPos[s] + 1.0) / (g_rPos[s] + g_rNeg[s] + 2.0);
}

// path measurement (bytes seen on S2/S3 ingress from S1, i.e. forward direction)
struct PathCounter
{
    uint64_t bytes = 0, pkts = 0, udp = 0, shortBytes = 0;
    std::set<uint32_t> srcs;
    std::set<uint64_t> flows;
    std::map<uint32_t, uint64_t> bytesBySrc;
};
static PathCounter g_pc[2];
static double g_utilShort[2] = {0, 0}, g_utilLong[2] = {0, 0};

// ------------------------------------------------------------------ task bookkeeping
struct TaskRec
{
    uint32_t id;
    int uav;
    bool fake;
    uint32_t sizeB;
    double cycles, deadline, created;
    int site = -1, path = -1, action = -1;
    Vec s, s2;
    bool hasS2 = false, done = false, rejected = false, rewardReady = false, lowTrust = false;
    bool unsafe = false, blocked = false;
    int optimal = -1;
    double latency = 0, upStart = -1, upEnd = -1, reward = 0, energy = 0, money = 0, repQ = 0;
};
static std::map<uint32_t, TaskRec> g_tasks;
static std::vector<uint32_t> g_awaitNext;
static uint32_t g_nextTaskId = 1;

// statistics
struct Stats
{
    double n = 0, rej = 0, latSum = 0, nDone = 0, eSum = 0, costSum = 0, moneySum = 0;
    double offl = 0, opt = 0, unsafe = 0, rewardSum = 0, work[5] = {0, 0, 0, 0, 0};
    double decUs = 0, decN = 0, lossSum = 0, lossN = 0, flowMods = 0;
};
static Stats g_win, g_eval;


// ------------------------------------------------------------------ forward declarations
class SecController;
class MiddlewareApp;
class ServerApp;
static Ptr<SecController> g_ctrl;
static Ptr<MiddlewareApp> g_mw;
static Ptr<ServerApp> g_srv[5];
static std::vector<double> g_uavRateEst; // measured uplink goodput per UAV (bit/s)
static std::vector<double> g_uavLocalBusy;

static void TaskFinished(uint32_t id, bool ok);

// =====================================================================================
//  SDN controller with DDQN agent, global detector and response module
// =====================================================================================
class SecController : public OFSwitch13Controller
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("SecController")
                                .SetParent<OFSwitch13Controller>()
                                .AddConstructor<SecController>();
        return tid;
    }

    uint64_t dpS1 = 0, dpS2 = 0, dpS3 = 0, dpS4 = 0;
    int ready = 0;

    void Cmd(uint64_t dp, const std::string& c)
    {
        DpctlExecute(dp, c);
        g_win.flowMods++;
        g_eval.flowMods++;
    }

    // per-task forwarding rule (both directions) on the chosen path
    void InstallTaskPath(uint16_t port, int path)
    {
        std::ostringstream a, b;
        a << "flow-mod cmd=add,table=0,prio=100,idle=10 eth_type=0x800,ip_proto=6,"
          << "ip_src=10.1.1.1,tcp_src=" << port << " apply:output=" << (path == 0 ? 2 : 3);
        b << "flow-mod cmd=add,table=0,prio=100,idle=10 eth_type=0x800,ip_proto=6,"
          << "ip_dst=10.1.1.1,tcp_dst=" << port << " apply:output=" << (path == 0 ? 1 : 2);
        Cmd(dpS1, a.str());
        Cmd(dpS4, b.str());
    }

    void SetAttackPath(int path)
    {
        Cmd(dpS1, "flow-mod cmd=add,table=0,prio=50 in_port=4 apply:output=" +
                      std::to_string(path == 0 ? 2 : 3));
    }

    // response module: drop everything from a source IP at the ingress switch
    void BlockSource(Ipv4Address ip)
    {
        std::ostringstream c;
        c << "flow-mod cmd=add,table=0,prio=300 eth_type=0x800,ip_src=" << ip;
        Cmd(dpS1, c.str());
    }

    void UnblockSource(Ipv4Address ip)
    {
        std::ostringstream c;
        c << "flow-mod cmd=del,table=0,prio=300 eth_type=0x800,ip_src=" << ip;
        Cmd(dpS1, c.str());
    }

  protected:
    void HandshakeSuccessful(Ptr<const RemoteSwitch> sw) override
    {
        uint64_t dp = sw->GetDpId();
        if (dp == dpS1)
        {
            // ports: 1 middleware, 2 -> S2, 3 -> S3, 4 attacker, 5 background source
            DpctlExecute(dp, "flow-mod cmd=add,table=0,prio=10 eth_type=0x800,ip_dst=10.1.1.1 apply:output=1");
            DpctlExecute(dp, "flow-mod cmd=add,table=0,prio=10 eth_type=0x800,ip_dst=10.1.1.5 apply:output=4");
            DpctlExecute(dp, "flow-mod cmd=add,table=0,prio=10 eth_type=0x800,ip_dst=10.1.1.6 apply:output=5");
            DpctlExecute(dp, "flow-mod cmd=add,table=0,prio=5 eth_type=0x800,ip_src=10.1.1.1 apply:output=2");
            DpctlExecute(dp, "flow-mod cmd=add,table=0,prio=20 in_port=5,eth_type=0x800,ip_proto=17,udp_dst=9001 apply:output=2");
            DpctlExecute(dp, "flow-mod cmd=add,table=0,prio=20 in_port=5,eth_type=0x800,ip_proto=17,udp_dst=9002 apply:output=3");
            DpctlExecute(dp, "flow-mod cmd=add,table=0,prio=50 in_port=4 apply:output=2");
            DpctlExecute(dp, "flow-mod cmd=add,table=0,prio=1 in_port=2 apply:output=1");
            DpctlExecute(dp, "flow-mod cmd=add,table=0,prio=1 in_port=3 apply:output=1");
        }
        else if (dp == dpS2 || dp == dpS3)
        {
            DpctlExecute(dp, "flow-mod cmd=add,table=0,prio=10 in_port=1 apply:output=2");
            DpctlExecute(dp, "flow-mod cmd=add,table=0,prio=10 in_port=2 apply:output=1");
        }
        else if (dp == dpS4)
        {
            // ports: 1 <- S2, 2 <- S3, 3 edge-1, 4 edge-2, 5 cloud, 6 background sink
            DpctlExecute(dp, "flow-mod cmd=add,table=0,prio=10 eth_type=0x800,ip_dst=10.1.1.2 apply:output=3");
            DpctlExecute(dp, "flow-mod cmd=add,table=0,prio=10 eth_type=0x800,ip_dst=10.1.1.3 apply:output=4");
            DpctlExecute(dp, "flow-mod cmd=add,table=0,prio=10 eth_type=0x800,ip_dst=10.1.1.4 apply:output=5");
            DpctlExecute(dp, "flow-mod cmd=add,table=0,prio=10 eth_type=0x800,ip_dst=10.1.2.2 apply:output=5");
            DpctlExecute(dp, "flow-mod cmd=add,table=0,prio=10 eth_type=0x800,ip_dst=10.1.1.7 apply:output=6");
            DpctlExecute(dp, "flow-mod cmd=add,table=0,prio=5 eth_type=0x800 apply:output=1");
        }
        ready++;
    }
};

// =====================================================================================
//  Edge / cloud server: receive task, FIFO CPU queue, return result
// =====================================================================================
class ServerApp : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid =
            TypeId("ServerApp").SetParent<Application>().AddConstructor<ServerApp>();
        return tid;
    }

    int site = 2;
    double busyUntil = 0, reportedQ = 0;
    bool compromised = false, compSeen = false; // compSeen: ground-truth label for the window

    double TrueQ() const
    {
        return std::max(0.0, busyUntil - Simulator::Now().GetSeconds());
    }

  private:
    struct Conn
    {
        uint64_t rx = 0;
        uint8_t hdr[32];
        uint32_t id = 0, size = 0;
        double cycles = 0;
        bool queued = false;
    };

    std::map<Ptr<Socket>, Conn> m_conns;
    Ptr<Socket> m_listen;

    void StartApplication() override
    {
        m_listen = Socket::CreateSocket(GetNode(), TcpSocketFactory::GetTypeId());
        m_listen->SetAttribute("SegmentSize", UintegerValue(8948));
        m_listen->Bind(InetSocketAddress(Ipv4Address::GetAny(), 6000));
        m_listen->Listen();
        m_listen->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                    MakeCallback(&ServerApp::Accept, this));
        Report();
    }

    void StopApplication() override
    {
    }

    // periodic state report to the controller (every 100 ms)
    void Report()
    {
        reportedQ = compromised ? 0.0 : TrueQ(); // a compromised server lies that it is idle
        compSeen = compSeen || compromised;
        Simulator::Schedule(MilliSeconds(100), &ServerApp::Report, this);
    }

    void Accept(Ptr<Socket> s, const Address&)
    {
        m_conns[s] = Conn();
        s->SetRecvCallback(MakeCallback(&ServerApp::Recv, this));
        s->SetCloseCallbacks(MakeCallback(&ServerApp::Closed, this),
                             MakeCallback(&ServerApp::Closed, this));
    }

    void Closed(Ptr<Socket> s)
    {
        s->Close();
        m_conns.erase(s);
    }

    void Recv(Ptr<Socket> s)
    {
        auto it = m_conns.find(s);
        if (it == m_conns.end())
        {
            return;
        }
        Conn& c = it->second;
        Ptr<Packet> p;
        while ((p = s->Recv()))
        {
            uint32_t n = p->GetSize();
            if (c.rx < 32)
            {
                uint32_t k = std::min<uint32_t>(32 - c.rx, n);
                p->CopyData(c.hdr + c.rx, k);
                if (c.rx + k == 32)
                {
                    std::memcpy(&c.id, c.hdr, 4);
                    std::memcpy(&c.size, c.hdr + 8, 4);
                    uint64_t cyc;
                    std::memcpy(&cyc, c.hdr + 12, 8);
                    c.cycles = double(cyc);
                }
            }
            c.rx += n;
        }
        if (!c.queued && c.rx >= 32 + uint64_t(c.size) && c.size > 0)
        {
            c.queued = true;
            DBG("server " << site << " got task " << c.id << " q " << TrueQ());
            double now = Simulator::Now().GetSeconds();
            double exec = c.cycles / CPU_HZ[site];
            if (TrueQ() + exec > MAXQ[site])
            { // admission control: queue full -> reject
                uint8_t st = 0;
                s->Send(Create<Packet>(&st, 1));
                s->Close();
                return;
            }
            if (compromised && U->GetValue() < g.dropProb)
            {
                busyUntil = std::max(busyUntil, now) + exec; // pretends to work, silently drops
                return;                                      // (connection left hanging)
            }
            busyUntil = std::max(busyUntil, now) + exec;
            Simulator::Schedule(Seconds(busyUntil - now), &ServerApp::Reply, this, s, c.id,
                                uint32_t(c.size * RESULT_RATIO), c.cycles);
        }
    }

    void Reply(Ptr<Socket> s, uint32_t id, uint32_t res, double cycles)
    {
        if (m_conns.find(s) == m_conns.end())
        {
            return;
        }
        auto t = g_tasks.find(id);
        if (t != g_tasks.end() && !t->second.fake)
        {
            g_eval.work[site] += cycles;
        }
        uint8_t st = 1;
        int r1 = s->Send(Create<Packet>(&st, 1));
        int r2 = s->Send(Create<Packet>(res));
        DBG("server " << site << " reply task " << id << " " << r1 << " " << r2);
        s->Close();
    }
};

// =====================================================================================
//  Middleware (edge gateway): screening, classification, decision request, relay
// =====================================================================================
static Vec BuildState(const TaskRec& t);

class MiddlewareApp : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid =
            TypeId("MiddlewareApp").SetParent<Application>().AddConstructor<MiddlewareApp>();
        return tid;
    }

    Ipv4Address srvAddr[5];
    double busyUntil = 0, reportedQ = 0;
    uint16_t nextPort = 20000;

    // local-detector window statistics
    struct UavWin
    {
        double n = 0, mb = 0, gc = 0, cpb = 0;
        bool fakeTruth = false;
    };
    struct SrvWin
    {
        double sent = 0, resolved = 0, timeouts = 0, repQ = 0, latRatio = 0, done = 0, silent = 0;
    };
    std::vector<UavWin> uw;
    SrvWin sw[5];

    double TrueQ() const
    {
        return std::max(0.0, busyUntil - Simulator::Now().GetSeconds());
    }

    // Called when a task misses its deadline. "Silent drop" = the server ACKed the whole task
    // (nothing left in the middleware's TCP send buffer) but never answered -> strong evidence
    // that the server is lying/dropping, as opposed to network congestion.
    void OnTimeout(const TaskRec& t)
    {
        if (t.site < 2)
            return;
        bool silent = false;
        auto it = m_sockOfTask.find(t.id);
        if (it != m_sockOfTask.end())
        {
            Ptr<Socket> s = it->second;
            // ...and the server had REPORTED it was free enough to finish well before the deadline
            bool shouldHaveMadeIt = t.repQ + t.cycles / CPU_HZ[t.site] < 0.6 * t.deadline;
            silent = shouldHaveMadeIt && m_srvRx.count(s) && m_srvRx[s] == 0 &&
                     s->GetTxAvailable() >= (4u << 20);
        }
        if (it != m_sockOfTask.end())
        {
            m_taskOfSock.erase(it->second);
            m_sockOfTask.erase(it);
        }
        if (silent)
            sw[t.site].silent++;
        if (M.mw && M.detect)
        { // consistency check in the trust engine: a silent drop is strong negative evidence
            SiteEvidence(t.site, 0.0, silent ? 3.0 : 0.5);
        }
    }

    std::map<uint32_t, Ptr<Socket>> m_sockOfTask;
    std::map<Ptr<Socket>, uint32_t> m_taskOfSock;

    void ForgetSock(Ptr<Socket> s)
    {
        auto it = m_taskOfSock.find(s);
        if (it != m_taskOfSock.end())
        {
            m_sockOfTask.erase(it->second);
            m_taskOfSock.erase(it);
        }
    }

  private:
    struct Conn
    {
        uint64_t rx = 0;
        uint8_t hdr[32];
        uint32_t id = 0;
        bool decided = false, forwarded = false, offload = false;
        Ptr<Socket> up;
    };

    std::map<Ptr<Socket>, Conn> m_uav;        // sockets from UAVs
    std::map<Ptr<Socket>, Ptr<Socket>> m_srv; // server socket -> UAV socket
    std::map<Ptr<Socket>, uint64_t> m_srvRx;
    Ptr<Socket> m_listen;

    void StartApplication() override
    {
        m_listen = Socket::CreateSocket(GetNode(), TcpSocketFactory::GetTypeId());
        m_listen->Bind(InetSocketAddress(Ipv4Address::GetAny(), 5000));
        m_listen->Listen();
        m_listen->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                    MakeCallback(&MiddlewareApp::Accept, this));
        Report();
    }

    void StopApplication() override
    {
    }

    void Report()
    {
        reportedQ = TrueQ();
        Simulator::Schedule(MilliSeconds(100), &MiddlewareApp::Report, this);
    }

    void Accept(Ptr<Socket> s, const Address&)
    {
        m_uav[s] = Conn();
        s->SetRecvCallback(MakeCallback(&MiddlewareApp::RecvUav, this));
        s->SetCloseCallbacks(MakeCallback(&MiddlewareApp::ClosedUav, this),
                             MakeCallback(&MiddlewareApp::ClosedUav, this));
    }

    void ClosedUav(Ptr<Socket> s)
    {
        s->Close();
        m_uav.erase(s);
    }

    void RecvUav(Ptr<Socket> s)
    {
        auto it = m_uav.find(s);
        if (it == m_uav.end())
        {
            return;
        }
        Conn& c = it->second;
        Ptr<Packet> p;
        while ((p = s->Recv()))
        {
            uint32_t n = p->GetSize();
            if (c.rx < 32)
            {
                uint32_t k = std::min<uint32_t>(32 - c.rx, n);
                p->CopyData(c.hdr + c.rx, k);
                if (c.rx + k == 32)
                {
                    std::memcpy(&c.id, c.hdr, 4);
                    Simulator::Schedule(Seconds(M.mw ? MW_PROC : 0.0001), &MiddlewareApp::Decide,
                                        this, s);
                }
            }
            c.rx += n;
        }
        auto t = g_tasks.find(c.id);
        if (c.offload && !c.forwarded && t != g_tasks.end() && c.rx >= 32 + t->second.sizeB)
        {
            c.forwarded = true;
            TaskRec& tr = t->second;
            tr.upEnd = Simulator::Now().GetSeconds();
            DBG("upload done task " << tr.id << " took " << tr.upEnd - tr.upStart);
            double up = std::max(1e-4, tr.upEnd - tr.upStart);
            g_uavRateEst[tr.uav] = 0.7 * g_uavRateEst[tr.uav] + 0.3 * (tr.sizeB * 8.0 / up);
            if (tr.site == 1)
            {
                RunLocal(s, tr);
            }
            else
            {
                Forward(s, tr);
            }
        }
    }

    // ---------------- offloading decision (asks the control plane) ----------------
    void Decide(Ptr<Socket> s)
    {
        auto it = m_uav.find(s);
        if (it == m_uav.end())
        {
            return;
        }
        Conn& c = it->second;
        auto ti = g_tasks.find(c.id);
        if (ti == g_tasks.end() || ti->second.done)
        {
            return;
        }
        TaskRec& t = ti->second;
        c.decided = true;
        // local detector statistics for this UAV
        uw[t.uav].n++;
        uw[t.uav].mb += t.sizeB / 1e6;
        uw[t.uav].gc += t.cycles / 1e9;
        uw[t.uav].cpb += t.cycles / (t.sizeB * 8.0);
        uw[t.uav].fakeTruth = uw[t.uav].fakeTruth || t.fake;
        // screening: middleware blocks UAVs flagged by the local detector
        if (M.mw && M.detect && g_trustUav[t.uav] < 0.5)
        {
            t.blocked = true;
            uint8_t d = 2;
            s->Send(Create<Packet>(&d, 1));
            g_reason["blocked"]++;
            TaskFinished(t.id, false);
            return;
        }
        // build MDP state, query agent (timed)
        Vec st = BuildState(t);
        auto t0 = std::chrono::steady_clock::now();
        int a = g_agent->Act(st, g_greedy);
        if (g_policy == "random")
        {
            a = int(U->GetInteger(0, g_actions.size() - 1));
        }
        bool useGreedy = g_policy == "greedy" || (g_policy == "mixed" && U->GetValue() < 0.7);
        if (useGreedy)
        { // reference heuristic: minimum estimated completion time from reported state
            double best = 1e18;
            for (size_t k = 0; k < g_actions.size(); ++k)
            {
                int site = g_actions[k].first, p = g_actions[k].second;
                double now = Simulator::Now().GetSeconds();
                double q = site == 0 ? std::max(0.0, g_uavLocalBusy[t.uav] - now)
                                     : (site == 1 ? reportedQ : g_srv[site]->reportedQ);
                double net = site == 0 ? 0 : t.sizeB * 8.0 / g_uavRateEst[t.uav];
                if (site >= 2)
                    net += t.sizeB * 8.0 / (PATH_CAP * (1.0 - std::min(0.95, g_utilShort[p < 0 ? t.uav % 2 : p])));
                if (site == 4)
                    net += 0.25;
                double v = q + net + t.cycles / CPU_HZ[site];
                if (v < best)
                {
                    best = v;
                    a = int(k);
                }
            }
        }
        // v3: trust-aware action masking - never send a task to a server whose trust is too low
        auto masked = [&](int k) {
            int site = g_actions[k].first;
            return site >= 2 && g_trustSite[site] < TRUST_MASK;
        };
        if (M.trust && M.mask && !useGreedy && g_policy != "random" && masked(a))
        {
            Vec q = g_agent->QValues(st);
            double best = -1e18;
            for (size_t k = 0; k < g_actions.size(); ++k)
            {
                if (!masked(int(k)) && q[k] > best)
                {
                    best = q[k];
                    a = int(k);
                }
            }
            g_reason["trust_masked"]++;
        }
        auto t1 = std::chrono::steady_clock::now();
        g_eval.decUs +=std::chrono::duration<double, std::micro>(t1 - t0).count();
        g_eval.decN++;
        // next-state bookkeeping for transitions
        for (uint32_t id : g_awaitNext)
        {
            auto o = g_tasks.find(id);
            if (o == g_tasks.end())
            {
                continue;
            }
            o->second.s2 = st;
            o->second.hasS2 = true;
            if (o->second.rewardReady && g_learn)
            {
                g_agent->Store({o->second.s, o->second.action, o->second.reward, st});
            }
            if (o->second.done)
            {
                g_tasks.erase(o);
            }
        }
        g_awaitNext.clear();
        g_awaitNext.push_back(t.id);
        if (g_learn)
        {
            for (int rep = 0; rep < 2; ++rep) // two gradient steps per decision
            {
                double l = g_agent->Learn();
                if (l >= 0)
                {
                    g_win.lossSum += l;
                    g_win.lossN++;
                }
            }
        }
        t.s = st;
        t.action = a;
        t.site = g_actions[a].first;
        t.path = g_actions[a].second;
        if (t.site >= 2 && t.path < 0)
        {
            t.path = t.uav % 2; // no SDN path selection: static hash path
        }
        if (t.site >= 2)
        {
            t.lowTrust = g_trustSite[t.site] < 0.5 || g_trustPath[t.path] < 0.5;
            bool pathAtt = g_ddos && Simulator::Now().GetSeconds() >= g_attackStart &&
                           t.path == g_ddosPath && !g_ddosMitigated;
            t.unsafe = (t.site == g_compSite) || pathAtt;
            int best = (g_utilShort[0] <= g_utilShort[1]) ? 0 : 1;
            t.optimal = (t.path == best) ? 1 : 0;
            sw[t.site].sent++;
            t.repQ = g_srv[t.site]->reportedQ;
        }
        DBG("decide task " << t.id << " uav " << t.uav << " site " << t.site << " path " << t.path << " size " << t.sizeB << " cyc " << t.cycles << " dl " << t.deadline);
        uint8_t d = (t.site == 0) ? 0 : 1;
        c.offload = (t.site != 0);
        s->Send(Create<Packet>(&d, 1));
        if (t.site == 0)
        {
            return; // UAV computes locally
        }
        if (t.site >= 2)
        {
            uint16_t port = nextPort++;
            if (nextPort > 60000)
            {
                nextPort = 20000;
            }
            g_ctrl->InstallTaskPath(port, t.path);
            m_portOf[t.id] = port;
        }
    }

    std::map<uint32_t, uint16_t> m_portOf;

    void RunLocal(Ptr<Socket> up, TaskRec& t)
    {
        double now = Simulator::Now().GetSeconds();
        double exec = t.cycles / CPU_HZ[1];
        if (TrueQ() + exec > MAXQ[1])
        {
            uint8_t st = 0;
            up->Send(Create<Packet>(&st, 1));
            return;
        }
        busyUntil = std::max(busyUntil, now) + exec;
        Simulator::Schedule(Seconds(busyUntil - now), &MiddlewareApp::LocalDone, this, up, t.id,
                            uint32_t(t.sizeB * RESULT_RATIO), t.cycles, t.fake);
    }

    void LocalDone(Ptr<Socket> up, uint32_t id, uint32_t res, double cycles, bool fake)
    {
        if (!fake)
        {
            g_eval.work[1] += cycles;
        }
        if (m_uav.find(up) == m_uav.end())
        {
            return;
        }
        uint8_t st = 1;
        up->Send(Create<Packet>(&st, 1));
        up->Send(Create<Packet>(res));
    }

    void Forward(Ptr<Socket> up, TaskRec& t)
    {
        Ptr<Socket> s = Socket::CreateSocket(GetNode(), TcpSocketFactory::GetTypeId());
        s->SetAttribute("SegmentSize", UintegerValue(8948)); // jumbo MSS on the wired side
        s->Bind(InetSocketAddress(Ipv4Address("10.1.1.1"), m_portOf[t.id]));
        m_portOf.erase(t.id);
        m_srv[s] = up;
        m_srvRx[s] = 0;
        m_srvExp[s] = 1 + uint64_t(t.sizeB * RESULT_RATIO);
        m_sockOfTask[t.id] = s;
        m_taskOfSock[s] = t.id;
        uint8_t hdr[32] = {0};
        std::memcpy(hdr, &t.id, 4);
        std::memcpy(hdr + 8, &t.sizeB, 4);
        uint64_t cyc = uint64_t(t.cycles);
        std::memcpy(hdr + 12, &cyc, 8);
        uint32_t size = t.sizeB;
        s->SetRecvCallback(MakeCallback(&MiddlewareApp::RecvSrv, this));
        s->SetCloseCallbacks(MakeCallback(&MiddlewareApp::ClosedSrv, this),
                             MakeCallback(&MiddlewareApp::ClosedSrv, this));
        Ptr<Packet> h = Create<Packet>(hdr, 32);
        s->SetConnectCallback(
            MakeBoundCallback(&MiddlewareApp::Connected, h, size),
            MakeNullCallback<void, Ptr<Socket>>());
        s->Connect(InetSocketAddress(srvAddr[t.site], 6000));
    }

    static void Connected(Ptr<Packet> hdr, uint32_t size, Ptr<Socket> s)
    {
        DBG("mw->server connected, sending " << size);
        s->Send(hdr);
        s->Send(Create<Packet>(size));
    }

    void ClosedSrv(Ptr<Socket> s)
    {
        ForgetSock(s);
        s->Close();
        m_srv.erase(s);
        m_srvRx.erase(s);
        m_srvExp.erase(s);
    }

    std::map<Ptr<Socket>, uint64_t> m_srvExp;

    void RecvSrv(Ptr<Socket> s)
    {
        auto it = m_srv.find(s);
        if (it == m_srv.end())
        {
            return;
        }
        Ptr<Packet> p;
        while ((p = s->Recv()))
        {
            uint8_t first = 1;
            if (m_srvRx[s] == 0)
            {
                p->CopyData(&first, 1);
            }
            m_srvRx[s] += p->GetSize();
            if (m_uav.find(it->second) != m_uav.end())
            {
                it->second->Send(p); // relay result bytes to the UAV
            }
            if (first == 0 || m_srvRx[s] >= m_srvExp[s])
            {
                ForgetSock(s);
                s->Close();
                m_srv.erase(s);
                m_srvRx.erase(s);
                m_srvExp.erase(s);
                return;
            }
        }
    }
};

// =====================================================================================
//  UAV application: Poisson task generation, request/decision handshake, upload, result
// =====================================================================================
class UavApp : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("UavApp").SetParent<Application>().AddConstructor<UavApp>();
        return tid;
    }

    int idx = 0;
    bool malicious = false;
    Ipv4Address mwAddr;

  private:
    struct Conn
    {
        uint32_t id;
        uint64_t rx = 0;
        int decision = -1;
        bool sentPayload = false;
    };

    std::map<Ptr<Socket>, Conn> m_c;
    std::map<uint32_t, Ptr<Socket>> m_byId;

    void StartApplication() override
    {
        ScheduleNext();
    }

    void StopApplication() override
    {
    }

    void ScheduleNext()
    {
        double rate = g.lambda;
        bool fakeNow = malicious && g_fake && Simulator::Now().GetSeconds() >= g_attackStart;
        if (fakeNow)
        {
            rate *= 3.0;
        }
        double dt = -std::log(1.0 - U->GetValue()) / rate;
        Simulator::Schedule(Seconds(dt), &UavApp::NewTask, this);
    }

    void NewTask()
    {
        ScheduleNext();
        bool fakeNow = malicious && g_fake && Simulator::Now().GetSeconds() >= g_attackStart;
        TaskRec t;
        t.id = g_nextTaskId++;
        t.uav = idx;
        t.fake = fakeNow;
        t.sizeB = uint32_t(U->GetValue(0.2e6, 1.2e6));
        t.cycles = t.sizeB * 8.0 * U->GetValue(300, 1300);
        if (t.fake)
        {
            t.cycles *= U->GetValue(5, 10);
        }
        t.deadline = U->GetValue(0.8, 3.0);
        t.created = Simulator::Now().GetSeconds();
        g_tasks[t.id] = t;
        Ptr<Socket> s = Socket::CreateSocket(GetNode(), TcpSocketFactory::GetTypeId());
        s->Bind();
        m_c[s] = Conn{t.id};
        m_byId[t.id] = s;
        s->SetRecvCallback(MakeCallback(&UavApp::Recv, this));
        s->SetConnectCallback(MakeCallback(&UavApp::Connected, this),
                              MakeCallback(&UavApp::Failed, this));
        s->Connect(InetSocketAddress(mwAddr, 5000));
        Simulator::Schedule(Seconds(t.deadline), &UavApp::Deadline, this, t.id);
    }

    void Connected(Ptr<Socket> s)
    {
        auto it = m_c.find(s);
        if (it == m_c.end())
        {
            return;
        }
        TaskRec& t = g_tasks[it->second.id];
        uint8_t hdr[32] = {0};
        std::memcpy(hdr, &t.id, 4);
        uint16_t u = idx;
        std::memcpy(hdr + 4, &u, 2);
        std::memcpy(hdr + 8, &t.sizeB, 4);
        uint64_t cyc = uint64_t(t.cycles);
        std::memcpy(hdr + 12, &cyc, 8);
        s->Send(Create<Packet>(hdr, 32));
    }

    void Failed(Ptr<Socket> s)
    {
    }

    void Recv(Ptr<Socket> s)
    {
        auto it = m_c.find(s);
        if (it == m_c.end())
        {
            return;
        }
        Conn& c = it->second;
        auto ti = g_tasks.find(c.id);
        if (ti == g_tasks.end())
        {
            return;
        }
        TaskRec& t = ti->second;
        Ptr<Packet> p;
        while ((p = s->Recv()))
        {
            uint32_t n = p->GetSize();
            uint8_t buf[2];
            uint32_t k = std::min<uint32_t>(n, 2);
            p->CopyData(buf, k);
            uint64_t before = c.rx;
            c.rx += n;
            if (before == 0)
            {
                c.decision = buf[0];
                if (c.decision == 2)
                { // blocked by the middleware
                    Finish(c.id, false);
                    return;
                }
                if (c.decision == 0)
                { // local execution on the UAV CPU
                    double now = Simulator::Now().GetSeconds();
                    double exec = t.cycles / CPU_HZ[0];
                    double q = std::max(0.0, g_uavLocalBusy[idx] - now);
                    if (q + exec > MAXQ[0])
                    {
                        g_reason["local_queue"]++;
                        Finish(c.id, false);
                        return;
                    }
                    g_uavLocalBusy[idx] = std::max(g_uavLocalBusy[idx], now) + exec;
                    Simulator::Schedule(Seconds(g_uavLocalBusy[idx] - now), &UavApp::LocalDone,
                                        this, c.id, t.cycles);
                    return;
                }
                // offload: upload the payload
                t.upStart = Simulator::Now().GetSeconds();
                s->Send(Create<Packet>(t.sizeB));
                if (n >= 2)
                {
                    if (buf[1] == 0)
                    {
                        Finish(c.id, false);
                        return;
                    }
                }
            }
            else if (before == 1)
            {
                if (buf[0] == 0)
                { // server/middleware rejected (queue full)
                    g_reason["server_queue"]++;
                    Finish(c.id, false);
                    return;
                }
            }
        }
        uint32_t res = uint32_t(t.sizeB * RESULT_RATIO);
        if (c.decision == 1 && c.rx >= 2 + uint64_t(res))
        {
            DBG("uav done task " << c.id);
            Finish(c.id, true);
        }
    }

    void LocalDone(uint32_t id, double cycles)
    {
        auto ti = g_tasks.find(id);
        if (ti != g_tasks.end() && !ti->second.done && !ti->second.fake)
        {
            g_eval.work[0] += cycles;
        }
        Finish(id, true);
    }

    void Deadline(uint32_t id)
    {
        auto ti = g_tasks.find(id);
        if (ti == g_tasks.end() || ti->second.done)
        {
            return;
        }
        DBG("TIMEOUT task " << id << " site " << ti->second.site);
        g_mw->OnTimeout(ti->second);
        g_reason["timeout_site" + std::to_string(ti->second.site)]++;
        Finish(id, false);
    }

    void Finish(uint32_t id, bool ok)
    {
        auto si = m_byId.find(id);
        if (si != m_byId.end())
        {
            Ptr<Socket> s = si->second;
            m_c.erase(s);
            m_byId.erase(si);
            s->Close();
        }
        auto ti = g_tasks.find(id);
        if (ti == g_tasks.end() || ti->second.done)
        {
            return;
        }
        TaskFinished(id, ok);
    }
};

// =====================================================================================
//  Background traffic and DDoS flooding (UDP)
// =====================================================================================
class UdpFlood : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("UdpFlood").SetParent<Application>().AddConstructor<UdpFlood>();
        return tid;
    }

    Ipv4Address dst;
    uint16_t dstPort = 9999;
    double rateBps = 0;
    uint32_t pktSize = 1000, nSrcPorts = 1;
    bool background = false;

  private:
    std::vector<Ptr<Socket>> m_s;
    uint32_t m_k = 0;

    void StartApplication() override
    {
        for (uint32_t i = 0; i < nSrcPorts; ++i)
        {
            Ptr<Socket> s = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
            s->Bind();
            s->Connect(InetSocketAddress(dst, dstPort));
            m_s.push_back(s);
        }
        Tick();
    }

    void StopApplication() override
    {
    }

    void Tick()
    {
        double rate = rateBps;
        if (!background)
        {
            bool on = g_ddos && Simulator::Now().GetSeconds() >= g_attackStart;
            rate = on ? rateBps : 0.0;
        }
        if (rate > 0)
        {
            m_s[m_k++ % m_s.size()]->Send(Create<Packet>(pktSize));
            Simulator::Schedule(Seconds(pktSize * 8.0 / rate), &UdpFlood::Tick, this);
        }
        else
        {
            Simulator::Schedule(MilliSeconds(50), &UdpFlood::Tick, this);
        }
    }
};

static std::vector<Ptr<UdpFlood>> g_bg;

static void
BackgroundWalk()
{
    for (auto& b : g_bg)
    {
        b->rateBps = std::clamp(b->rateBps + U->GetValue(-4e6, 4e6), 10e6, 45e6);
    }
    Simulator::Schedule(Seconds(1), &BackgroundWalk);
}

// =====================================================================================
//  Path monitoring (switch port statistics) - input of the global detector and the state
// =====================================================================================
static void
PathRx(int path, Ptr<const Packet> pkt)
{
    Ptr<Packet> p = pkt->Copy();
    EthernetHeader eth;
    if (!p->RemoveHeader(eth) || eth.GetLengthType() != 0x0800)
    {
        return;
    }
    Ipv4Header ip;
    p->RemoveHeader(ip);
    PathCounter& c = g_pc[path];
    uint32_t sz = pkt->GetSize();
    c.bytes += sz;
    c.shortBytes += sz;
    c.pkts++;
    uint32_t src = ip.GetSource().Get();
    c.srcs.insert(src);
    c.bytesBySrc[src] += sz;
    uint16_t sp = 0, dp = 0;
    if (ip.GetProtocol() == 17)
    {
        c.udp++;
        UdpHeader u;
        p->PeekHeader(u);
        sp = u.GetSourcePort();
        dp = u.GetDestinationPort();
    }
    else if (ip.GetProtocol() == 6)
    {
        TcpHeader t;
        p->PeekHeader(t);
        sp = t.GetSourcePort();
        dp = t.GetDestinationPort();
    }
    c.flows.insert((uint64_t(src) << 32) ^ (uint64_t(sp) << 16) ^ dp ^
                   (uint64_t(ip.GetDestination().Get()) << 20));
}

static void
UtilSample()
{
    for (int p = 0; p < 2; ++p)
    {
        g_utilShort[p] = std::min(1.0, g_pc[p].shortBytes * 8.0 / (PATH_CAP * 0.1));
        g_utilLong[p] = 0.8 * g_utilLong[p] + 0.2 * g_utilShort[p];
        g_pc[p].shortBytes = 0;
    }
    Simulator::Schedule(MilliSeconds(100), &UtilSample);
}

// =====================================================================================
//  MDP state, reward
// =====================================================================================
static Vec
BuildState(const TaskRec& t)
{
    Vec s;
    s.push_back(t.sizeB / 1.2e6);
    s.push_back(t.cycles / 2e10);
    s.push_back(t.deadline / 3.0);
    s.push_back(M.mw ? double(t.deadline < 1.3) : 0.0); // classifier: delay-sensitive
    s.push_back(M.mw ? double(t.cycles > 6e9) : 0.0);   // classifier: compute-heavy
    s.push_back(g_uavRateEst[t.uav] / 300e6);
    double now = Simulator::Now().GetSeconds();
    s.push_back(std::max(0.0, g_uavLocalBusy[t.uav] - now) / MAXQ[0]);
    s.push_back(g_mw->reportedQ / MAXQ[1]);
    for (int k = 2; k <= 4; ++k)
    {
        s.push_back(g_srv[k]->reportedQ / MAXQ[k]);
    }
    for (int p = 0; p < 2; ++p)
    {
        s.push_back(M.paths ? g_utilShort[p] : 0.0);
        s.push_back(M.paths ? g_utilLong[p] : 0.0);
    }
    for (int k = 2; k <= 4; ++k)
    {
        s.push_back(M.trust ? g_trustSite[k] : 0.0);
    }
    for (int p = 0; p < 2; ++p)
    {
        s.push_back(M.trust ? g_trustPath[p] : 0.0);
    }
    return s; // 20 values
}

static void
TaskFinished(uint32_t id, bool ok)
{
    auto ti = g_tasks.find(id);
    if (ti == g_tasks.end())
    {
        return;
    }
    TaskRec& t = ti->second;
    t.done = true;
    double now = Simulator::Now().GetSeconds();
    t.rejected = !ok;
    t.latency = ok ? now - t.created : t.deadline;
    // energy model
    double e = 0, money = 0;
    if (t.site == 0)
    {
        e = KAPPA_E[0] * t.cycles;
    }
    else if (t.site > 0)
    {
        double up = (t.upEnd > 0 && t.upStart > 0) ? t.upEnd - t.upStart
                                                   : t.sizeB * 8.0 / g_uavRateEst[t.uav];
        e = RADIO_W * up + KAPPA_E[t.site] * t.cycles;
        money = t.cycles / 1e9 * COST_GC[t.site] + (t.site == 4 ? t.sizeB / 1e6 * 0.002 : 0.0);
    }
    t.energy = e;
    t.money = money;
    if (t.site >= 2 && g_mw)
    { // middleware observes the outcome of every task it forwarded (consistency evidence)
        auto& w = g_mw->sw[t.site];
        w.resolved++;
        w.repQ += t.repQ;
        if (t.rejected)
            w.timeouts++;
        else
        {
            w.done++;
            w.latRatio += t.latency / t.deadline;
            if (M.mw && M.detect)
            {
                SiteEvidence(t.site, 1.0, 0.0); // answered task = positive evidence
            }
        }
    }
    double cost = t.latency + 0.5 * e + 0.5 * money;
    // reward of the MDP
    double r = -(t.latency + 0.5 * e / 2.0 + 0.5 * money * 5.0);
    if (t.rejected)
    {
        r -= 3.0;
    }
    if (M.secReward && t.lowTrust)
    {
        r -= 2.0;
    }
    t.reward = r;
    if (t.action >= 0 && g_learn)
    {
        if (t.hasS2)
        {
            g_agent->Store({t.s, t.action, r, t.s2});
        }
        else
        {
            t.rewardReady = true;
        }
    }
    // statistics (legit tasks only)
    if (!t.fake)
    {
        for (Stats* st : {&g_win, &g_eval})
        {
            if (st == &g_eval && t.created < g.warmup)
            {
                continue;
            }
            st->n++;
            st->rewardSum += r;
            st->rej += t.rejected;
            if (!t.rejected)
            {
                st->nDone++;
                st->latSum += t.latency;
            }
            st->eSum += e;
            st->moneySum += money;
            st->costSum += cost;
            if (t.site >= 2)
            {
                st->offl++;
                st->unsafe += t.unsafe;
                st->opt += (t.optimal == 1);
            }
        }
    }
    // free memory of old tasks
    if (!t.hasS2 && t.rewardReady)
    {
        return;
    }
    g_tasks.erase(ti);
}

// =====================================================================================
//  Detector: window features, labels, online use (trust + response)
// =====================================================================================
static std::ofstream g_detCsv;
static double g_staticMu = 0, g_staticSd = 1; // conventional IDS: normal packet-rate statistics
static const double SRV_EW = 0.7;            // server-feature memory per 1-s window
static MiddlewareApp::SrvWin g_srvEw[5];     // sliding-horizon server statistics
static Vec g_pathStatsMu, g_pathStatsSd;
static int g_pathAlarmRun[2] = {0, 0};
static Ipv4Address g_blocked;
static bool g_hasBlock = false;
static double g_firstAlarm[3] = {-1, -1, -1}; // ddos, fake, comp (online, per window)

static void
DetectorTick()
{
    double now = Simulator::Now().GetSeconds();
    bool attackOn = now >= g_attackStart;
    // ---- path windows (global detector at the controller)
    for (int p = 0; p < 2; ++p)
    {
        PathCounter& c = g_pc[p];
        Vec f{double(c.pkts), c.bytes * 8.0 / 1e6, c.pkts ? double(c.bytes) / c.pkts : 0.0,
              double(c.srcs.size()), c.pkts ? double(c.udp) / c.pkts : 0.0,
              double(c.flows.size())};
        int label = c.bytesBySrc.count(Ipv4Address("10.1.1.5").Get()) ? 1 : 0; // attacker traffic seen
        if (g_detCsv.is_open())
        {
            g_detCsv << now << ',' << g_windowIdx << ',' << g_curScenario << ",0," << p << ','
                     << label << ',' << g_attackStart;
            for (double x : f)
                g_detCsv << ',' << x;
            g_detCsv << '\n';
            g_detCsv.flush();
        }
        else if (M.detect)
        {
            bool al = M.staticIds ? f[0] > g_staticMu + 3.0 * g_staticSd : g_aePath.Alarm(f);
            g_trustPath[p] = 0.7 * g_trustPath[p] + 0.3 * (al ? 0.0 : 1.0);
            if (al && label && g_firstAlarm[0] < 0)
            {
                g_firstAlarm[0] = now - g_attackStart;
            }
            g_pathAlarmRun[p] = al ? g_pathAlarmRun[p] + 1 : 0;
            if (M.response && g_pathAlarmRun[p] >= 2 && !g_hasBlock)
            { // response module: block the top-volume source on the flagged path
                uint32_t top = 0;
                uint64_t mx = 0;
                for (auto& kv : c.bytesBySrc)
                {
                    if (kv.second > mx && kv.first != Ipv4Address("10.1.1.1").Get() &&
                        kv.first != Ipv4Address("10.1.1.6").Get())
                    {
                        mx = kv.second;
                        top = kv.first;
                    }
                }
                if (top)
                {
                    g_blocked = Ipv4Address(top);
                    g_hasBlock = true;
                    g_ctrl->BlockSource(g_blocked);
                    if (g_blocked == Ipv4Address("10.1.1.5"))
                    {
                        g_ddosMitigated = true;
                    }
                }
            }
        }
        c = PathCounter{};
    }
    // ---- UAV windows (local detector at the middleware)
    for (uint32_t u = 0; u < g.nUav; ++u)
    {
        auto& w = g_mw->uw[u];
        if (w.n > 0)
        {
            Vec f{w.n, w.mb / w.n, w.gc / w.n, w.cpb / w.n};
            int label = w.fakeTruth ? 1 : 0;
            if (g_detCsv.is_open())
            {
                g_detCsv << now << ',' << g_windowIdx << ',' << g_curScenario << ",1," << u << ','
                         << label << ',' << g_attackStart;
                for (double x : f)
                    g_detCsv << ',' << x;
                g_detCsv << ",0,0\n";
            }
            else if (M.detect && M.mw)
            {
                bool al = g_aeUav.Alarm(f);
                g_trustUav[u] = 0.5 * g_trustUav[u] + 0.5 * (al ? 0.0 : 1.0);
                if (al && label && g_firstAlarm[1] < 0)
                {
                    g_firstAlarm[1] = now - g_attackStart;
                }
            }
        }
        w = MiddlewareApp::UavWin{};
    }
    // ---- server windows (local detector at the middleware: consistency of reports)
    for (int s = 2; s <= 4; ++s)
    {
        // v3: evidence about a server arrives late (a dropped task only times out at its
        // deadline, 0.8-3 s later), so server features use a sliding horizon (EWMA, ~3 s memory)
        auto& raw = g_mw->sw[s];
        auto& w = g_srvEw[s];
        w.sent = SRV_EW * w.sent + raw.sent;
        w.resolved = SRV_EW * w.resolved + raw.resolved;
        w.timeouts = SRV_EW * w.timeouts + raw.timeouts;
        w.repQ = SRV_EW * w.repQ + raw.repQ;
        w.latRatio = SRV_EW * w.latRatio + raw.latRatio;
        w.done = SRV_EW * w.done + raw.done;
        w.silent = SRV_EW * w.silent + raw.silent;
        raw = MiddlewareApp::SrvWin{};
        // fade old trust evidence towards the prior (one step per second)
        g_rPos[s] = TRUST_PRIOR_R + TRUST_FADE * (g_rPos[s] - TRUST_PRIOR_R);
        g_rNeg[s] *= TRUST_FADE;
        SiteEvidence(s, 0.0, 0.0);
        int label = g_srv[s]->compSeen ? 1 : 0;
        g_srv[s]->compSeen = false;
        if (w.resolved > 0.5)
        {
            Vec f{w.resolved, w.timeouts / w.resolved, w.silent / w.resolved, w.repQ / w.resolved,
                  w.done > 0.1 ? w.latRatio / w.done : 1.0};
            if (g_detCsv.is_open())
            {
                g_detCsv << now << ',' << g_windowIdx << ',' << g_curScenario << ",2," << s << ','
                         << label << ',' << g_attackStart;
                for (double x : f)
                    g_detCsv << ',' << x;
                g_detCsv << ",0\n";
            }
            else if (M.detect && M.mw)
            {
                bool al = g_aeSrv.Alarm(f);
                if (al)
                {
                    SiteEvidence(s, 0.0, 2.0); // detector alarm = negative evidence
                }
                if (al && label && g_firstAlarm[2] < 0)
                {
                    g_firstAlarm[2] = std::max(0.0, now - g_attackStart);
                }
            }
        }
    }
    Simulator::Schedule(Seconds(1), &DetectorTick);
}

// =====================================================================================
//  Scenario windows (episodes)
// =====================================================================================
static std::ofstream g_trainLog;

static void
SetScenario(const std::string& sc, double start)
{
    g_curScenario = sc;
    g_ddos = (sc == "DDoS on path" || sc == "Mixed");
    g_fake = (sc == "Fake tasks" || sc == "Mixed");
    bool comp = (sc == "Compromised edge" || sc == "Mixed");
    g_ddosPath = int(U->GetInteger(0, 1));
    g_compSite = comp ? int(U->GetInteger(2, 3)) : -1;
    g_attackStart = start;
    g_ddosMitigated = false;
    for (int s = 2; s <= 4; ++s)
    {
        g_srv[s]->compromised = (s == g_compSite);
        g_rPos[s] = TRUST_PRIOR_R;
        g_rNeg[s] = 0.0;
        SiteEvidence(s, 0.0, 0.0);
        g_srvEw[s] = MiddlewareApp::SrvWin{};
    }
    g_trustPath[0] = g_trustPath[1] = 1.0;
    std::fill(g_trustUav.begin(), g_trustUav.end(), 1.0);
    g_pathAlarmRun[0] = g_pathAlarmRun[1] = 0;
    if (g_hasBlock)
    {
        g_ctrl->UnblockSource(g_blocked);
        g_hasBlock = false;
    }
    if (g_ddos)
    {
        g_ctrl->SetAttackPath(g_ddosPath);
    }
    g_firstAlarm[0] = g_firstAlarm[1] = g_firstAlarm[2] = -1;
}

static std::string
RandomScenario()
{
    double r = U->GetValue();
    if (r < 0.4)
        return "No attack";
    if (r < 0.55)
        return "DDoS on path";
    if (r < 0.70)
        return "Compromised edge";
    if (r < 0.85)
        return "Fake tasks";
    return "Mixed";
}

static void
WindowBoundary()
{
    double now = Simulator::Now().GetSeconds();
    if (g_windowIdx > 0 && g_trainLog.is_open())
    {
        g_trainLog << g_windowIdx << ',' << g_curScenario << ',' << g_win.rewardSum << ','
                   << g_win.n << ',' << (g_win.n ? g_win.rej / g_win.n : 0) << ','
                   << (g_win.nDone ? g_win.latSum / g_win.nDone : 0) << ',' << g_agent->eps << ','
                   << (g_win.lossN ? g_win.lossSum / g_win.lossN : -1) << '\n';
        g_trainLog.flush();
        if (g_windowIdx % 20 == 0)
        {
            std::cout << "[" << g.method << "] window " << g_windowIdx << "/" << g.windows
                      << " reward=" << g_win.rewardSum << " rej="
                      << (g_win.n ? g_win.rej / g_win.n : 0) << " eps=" << g_agent->eps
                      << std::endl;
        }
    }
    g_win = Stats{};
    g_windowIdx++;
    if (g_learn)
    {
        double frac = double(g_windowIdx) / (0.65 * g.windows);
        g_agent->eps = std::max(0.02, 1.0 - 0.98 * frac);
    }
    // v3: episodic reset - every episode starts with empty compute queues, so the backlog left
    // by an attack in one episode does not leak into the next one
    for (int s = 2; s <= 4; ++s)
    {
        g_srv[s]->busyUntil = std::min(g_srv[s]->busyUntil, now);
    }
    g_mw->busyUntil = std::min(g_mw->busyUntil, now);
    for (auto& b : g_uavLocalBusy)
    {
        b = std::min(b, now);
    }
    SetScenario(RandomScenario(), now + 1.0);
    Simulator::Schedule(Seconds(g.window), &WindowBoundary);
}

// =====================================================================================
//  Detector training / evaluation from CSV  (--mode=traindet)
// =====================================================================================
struct Row
{
    double t, start;
    int win, etype, eid, label;
    std::string sc;
    Vec f;
};

static std::vector<Row>
ReadCsv(const std::string& fn)
{
    std::vector<Row> rows;
    std::ifstream is(fn);
    std::string line;
    while (std::getline(is, line))
    {
        std::stringstream ss(line);
        std::string tok;
        std::vector<std::string> v;
        while (std::getline(ss, tok, ','))
            v.push_back(tok);
        if (v.size() < 8)
            continue;
        Row r;
        r.t = std::stod(v[0]);
        r.win = std::stoi(v[1]);
        r.sc = v[2];
        r.etype = std::stoi(v[3]);
        r.eid = std::stoi(v[4]);
        r.label = std::stoi(v[5]);
        r.start = std::stod(v[6]);
        int nf = r.etype == 0 ? 6 : (r.etype == 1 ? 4 : 5);
        if ((int)v.size() < 7 + nf)
            continue; // incomplete line
        for (int i = 0; i < nf; ++i)
            r.f.push_back(std::stod(v[7 + i]));
        rows.push_back(r);
    }
    return rows;
}

static void
TrainDetector()
{
    auto tr = ReadCsv(g.out + "/det_train.csv");
    auto te = ReadCsv(g.out + "/det_test.csv");
    secnn::Autoencoder* ae[3] = {&g_aePath, &g_aeUav, &g_aeSrv};
    int bott[3] = {2, 2, 2};
    double mu[3] = {0, 0, 0}, sd[3] = {0, 0, 0}; // static-threshold baseline on primary feature
    int prim[3] = {0, 0, 2};                     // pkt/s, req/s, silent-drop ratio
    for (int e = 0; e < 3; ++e)
    {
        std::vector<Vec> normal;
        for (auto& r : tr)
            if (r.etype == e && r.label == 0)
                normal.push_back(r.f);
        std::shuffle(normal.begin(), normal.end(), std::mt19937(7));
        size_t k = normal.size() * 7 / 10;
        std::vector<Vec> fit(normal.begin(), normal.begin() + k), val(normal.begin() + k, normal.end());
        ae[e]->Fit(fit, bott[e], 150, 11 + e);
        ae[e]->Calibrate(val, 99.0);
        ae[e]->Save(g.models + "/ae_" + std::to_string(e) + ".txt");
        for (auto& x : fit)
            mu[e] += x[prim[e]] / fit.size();
        for (auto& x : fit)
            sd[e] += std::pow(x[prim[e]] - mu[e], 2) / fit.size();
        sd[e] = std::sqrt(sd[e]);
        std::cout << "AE " << e << ": trained on " << fit.size() << " normal windows, threshold "
                  << ae[e]->threshold << std::endl;
    }
    // evaluate three detectors on the test windows
    const char* names[3] = {"Static threshold", "Local AE (middleware only)", "Two-tier AE (proposed)"};
    std::ofstream res(g.out + "/detector_results.csv");
    res << "detector,accuracy,precision,recall,f1,fpr,tp,fp,fn,tn,recall_ddos,recall_fake,recall_comp\n";
    std::ofstream roc(g.out + "/detector_scores.csv");
    roc << "etype,label,score\n";
    for (int d = 0; d < 3; ++d)
    {
        double tp = 0, fp = 0, fn = 0, tn = 0, rec[3] = {0, 0, 0}, pos[3] = {0, 0, 0};
        for (auto& r : te)
        {
            bool pred;
            if (d == 0)
                pred = r.f[prim[r.etype]] > mu[r.etype] + 3 * sd[r.etype];
            else if (d == 1)
                pred = (r.etype == 0) ? false : ae[r.etype]->Alarm(r.f); // no global tier
            else
                pred = ae[r.etype]->Alarm(r.f);
            if (d == 2)
                roc << r.etype << ',' << r.label << ',' << ae[r.etype]->Score(r.f) / ae[r.etype]->threshold << '\n';
            if (r.label)
            {
                pos[r.etype]++;
                rec[r.etype] += pred;
                pred ? tp++ : fn++;
            }
            else
            {
                pred ? fp++ : tn++;
            }
        }
        double acc = (tp + tn) / (tp + tn + fp + fn), prec = tp / std::max(1.0, tp + fp),
               recl = tp / std::max(1.0, tp + fn), f1 = 2 * prec * recl / std::max(1e-9, prec + recl);
        res << names[d] << ',' << acc << ',' << prec << ',' << recl << ',' << f1 << ','
            << fp / std::max(1.0, fp + tn) << ',' << tp << ',' << fp << ',' << fn << ',' << tn << ','
            << rec[0] / std::max(1.0, pos[0]) << ',' << rec[1] / std::max(1.0, pos[1]) << ','
            << rec[2] / std::max(1.0, pos[2]) << '\n';
        std::cout << names[d] << " acc=" << acc << " P=" << prec << " R=" << recl << " F1=" << f1
                  << std::endl;
    }
    // detection time: first alarm (two-tier) after attack start within each attacked window
    std::map<std::pair<int, int>, double> first, start; // (window, etype)
    for (auto& r : te)
    {
        if (!r.label)
            continue;
        auto key = std::make_pair(r.win, r.etype);
        if (!start.count(key))
            start[key] = r.start;
        // window rows close at r.t and cover (r.t-1, r.t]; ignore rows ending before the attack
        // starts (leftover traffic of the previous episode carried into the new episode id)
        if (!first.count(key) && r.t > r.start && ae[r.etype]->Alarm(r.f))
            first[key] = r.t;
    }
    std::ofstream dt(g.out + "/detection_time.csv");
    dt << "attack,mean_detection_time_s,detected_windows,attacked_windows\n";
    const char* an[3] = {"DDoS on path", "Fake tasks", "Compromised node"};
    for (int e = 0; e < 3; ++e)
    {
        double sum = 0, n = 0, tot = 0;
        for (auto& kv : start)
        {
            if (kv.first.second != e)
                continue;
            tot++;
            if (first.count(kv.first))
            {
                sum += std::max(0.0, first[kv.first] - kv.second);
                n++;
            }
        }
        dt << an[e] << ',' << (n ? sum / n : -1) << ',' << n << ',' << tot << '\n';
    }
}

// conventional IDS baseline: 3-sigma rule on the packet rate of normal path windows
static void
SaveStaticIds()
{
    auto tr = ReadCsv(g.out + "/det_train.csv");
    double n = 0, mu = 0, sq = 0;
    for (auto& r : tr)
        if (r.etype == 0 && r.label == 0)
        {
            n++;
            mu += r.f[0];
            sq += r.f[0] * r.f[0];
        }
    mu /= std::max(1.0, n);
    double sd = std::sqrt(std::max(1e-9, sq / std::max(1.0, n) - mu * mu));
    std::ofstream(g.models + "/static_ids.txt") << mu << ' ' << sd << '\n';
    std::cout << "static IDS: packet-rate mean " << mu << " sd " << sd << std::endl;
}

// =====================================================================================
//  main: build the network
// =====================================================================================
int
main(int argc, char* argv[])
{
    CommandLine cmd;
    cmd.AddValue("mode", "collect | traindet | train | eval", g.mode);
    cmd.AddValue("method", "DQNEC | DDQNEC | SDQNEC | Proposed | Proposed_wo_*", g.method);
    cmd.AddValue("scenario", "eval scenario", g.scenario);
    cmd.AddValue("nUav", "number of UAVs", g.nUav);
    cmd.AddValue("lambda", "task arrival rate per UAV (tasks/s)", g.lambda);
    cmd.AddValue("seed", "random seed/run", g.seed);
    cmd.AddValue("windows", "training episodes (windows)", g.windows);
    cmd.AddValue("window", "episode length (s)", g.window);
    cmd.AddValue("simTime", "evaluation time (s)", g.simTime);
    cmd.AddValue("out", "results directory", g.out);
    cmd.AddValue("models", "models directory", g.models);
    cmd.AddValue("policy", "agent | greedy | random | mixed", g_policy);
    cmd.AddValue("ddosMbps", "DDoS flood rate (Mbit/s)", g.ddosMbps);
    cmd.AddValue("maskTh", "trust threshold for action masking (default 0.4)", TRUST_MASK);
    cmd.AddValue("dropProb", "fraction of tasks a compromised server silently drops", g.dropProb);
    uint32_t trainSeed = 1;
    cmd.AddValue("trainSeed", "which trained model (training seed) to evaluate", trainSeed);
    cmd.Parse(argc, argv);
    for (auto& ch : g.scenario) // ns-3 CommandLine stops at spaces: use '_' on the command line
        if (ch == '_')
            ch = ' ';
    SystemPath::MakeDirectories(g.out);
    SystemPath::MakeDirectories(g.models);

    if (g.mode == "traindet")
    {
        TrainDetector();
        SaveStaticIds();
        return 0;
    }
    if (g.mode == "staticdet")
    {
        SaveStaticIds();
        return 0;
    }

    RngSeedManager::SetSeed(12345);
    RngSeedManager::SetRun(g.seed);
    U = CreateObject<UniformRandomVariable>();
    M = GetMethod(g.mode == "collect" ? "Heuristic" : g.method);
    BuildActions();
    g_agent = std::make_unique<secnn::DqnAgent>(20, g_actions.size(), M.ddqn, g.seed * 7 + 3);
    g_learn = (g.mode == "train");
    g_greedy = (g.mode == "eval");
    if (g.method == "Greedy")
        g_policy = "greedy";
    if (g.method == "Random")
        g_policy = "random";
    const std::string modelFile =
        g.models + "/" + g.method + "_s" + std::to_string(g.mode == "train" ? g.seed : trainSeed) + ".dqn";
    if (g.mode == "eval" && g_policy == "agent")
    {
        std::ifstream test(modelFile);
        NS_ABORT_MSG_IF(!test.good(), "model not found: " << modelFile);
        g_agent->Load(modelFile);
        g_agent->eps = 0.0;
    }
    if (g.mode == "collect")
    {
        g_agent->eps = 1.0; // random exploration policy for data collection
    }
    if (M.detect && g.mode != "collect")
    {
        bool ok = g_aePath.Load(g.models + "/ae_0.txt") && g_aeUav.Load(g.models + "/ae_1.txt") &&
                  g_aeSrv.Load(g.models + "/ae_2.txt");
        NS_ABORT_MSG_IF(!ok, "detector models missing: run --mode=collect and --mode=traindet first");
    }
    if (M.staticIds)
    {
        std::ifstream st(g.models + "/static_ids.txt");
        NS_ABORT_MSG_IF(!(st >> g_staticMu >> g_staticSd), "static_ids.txt missing: run --mode=staticdet");
    }
    g_trustUav.assign(g.nUav, 1.0);
    g_uavRateEst.assign(g.nUav, 100e6);
    g_uavLocalBusy.assign(g.nUav, 0.0);

    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(4 << 20));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(4 << 20));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1448));
    Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(32));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(2));

    // ---------------- nodes ----------------
    NodeContainer uavs;
    uavs.Create(g.nUav);
    Ptr<Node> mw = CreateObject<Node>(), ctrl = CreateObject<Node>();
    NodeContainer sw;
    sw.Create(4);
    NodeContainer hosts; // 0 edge-1, 1 edge-2, 2 cloud gateway, 3 attacker, 4 bg source, 5 bg sink
    hosts.Create(6);
    Ptr<Node> cloud = CreateObject<Node>();

    // ---------------- Wi-Fi access (UAV <-> middleware AP) ----------------
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211ac);
    wifi.SetRemoteStationManager("ns3::IdealWifiManager");
    YansWifiChannelHelper chan;
    chan.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    chan.AddPropagationLoss("ns3::LogDistancePropagationLossModel", "Exponent", DoubleValue(2.2),
                            "ReferenceLoss", DoubleValue(46.67));
    YansWifiPhyHelper phy;
    phy.SetChannel(chan.Create());
    phy.Set("ChannelSettings", StringValue("{42, 80, BAND_5GHZ, 0}"));
    phy.Set("TxPowerStart", DoubleValue(20.0));
    phy.Set("TxPowerEnd", DoubleValue(20.0));
    WifiMacHelper mac;
    Ssid ssid("uav-net");
    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid), "BE_MaxAmpduSize", UintegerValue(1048575), "BE_MaxAmsduSize", UintegerValue(11398));
    NetDeviceContainer staDev = wifi.Install(phy, mac, uavs);
    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid), "BE_MaxAmpduSize", UintegerValue(1048575), "BE_MaxAmsduSize", UintegerValue(11398));
    NetDeviceContainer apDev = wifi.Install(phy, mac, mw);

    MobilityHelper mob;
    mob.SetPositionAllocator("ns3::RandomBoxPositionAllocator", "X",
                             StringValue("ns3::UniformRandomVariable[Min=0|Max=200]"), "Y",
                             StringValue("ns3::UniformRandomVariable[Min=0|Max=200]"), "Z",
                             StringValue("ns3::ConstantRandomVariable[Constant=50]"));
    mob.SetMobilityModel("ns3::RandomWalk2dMobilityModel", "Bounds",
                         RectangleValue(Rectangle(0, 200, 0, 200)), "Speed",
                         StringValue("ns3::UniformRandomVariable[Min=5|Max=15]"), "Mode",
                         StringValue("Time"), "Time", StringValue("2s"));
    mob.Install(uavs);
    MobilityHelper fixed;
    Ptr<ListPositionAllocator> pa = CreateObject<ListPositionAllocator>();
    pa->Add(Vector(100, 100, 10));
    fixed.SetPositionAllocator(pa);
    fixed.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    fixed.Install(mw);

    // ---------------- OpenFlow data plane ----------------
    CsmaHelper fast, path, wan;
    fast.SetDeviceAttribute("Mtu", UintegerValue(9000)); // jumbo frames on the wired backbone
    path.SetDeviceAttribute("Mtu", UintegerValue(9000));
    wan.SetDeviceAttribute("Mtu", UintegerValue(9000));
    fast.SetChannelAttribute("DataRate", DataRateValue(DataRate("1Gbps")));
    fast.SetChannelAttribute("Delay", TimeValue(MicroSeconds(1)));
    path.SetChannelAttribute("DataRate", DataRateValue(DataRate("100Mbps")));
    path.SetChannelAttribute("Delay", TimeValue(MicroSeconds(5)));
    wan.SetChannelAttribute("DataRate", DataRateValue(DataRate("1Gbps")));
    wan.SetChannelAttribute("Delay", TimeValue(MicroSeconds(1)));

    NetDeviceContainer swPorts[4], hostDevs;
    auto link = [&](CsmaHelper& h, Ptr<Node> a, Ptr<Node> b) {
        return h.Install(NodeContainer(a, b));
    };
    // S1: 1 middleware, 2 S2, 3 S3, 4 attacker, 5 bg source
    NetDeviceContainer l;
    l = link(fast, mw, sw.Get(0));
    hostDevs.Add(l.Get(0));
    swPorts[0].Add(l.Get(1));
    NetDeviceContainer l12 = link(path, sw.Get(0), sw.Get(1));
    NetDeviceContainer l13 = link(path, sw.Get(0), sw.Get(2));
    swPorts[0].Add(l12.Get(0));
    swPorts[0].Add(l13.Get(0));
    l = link(fast, hosts.Get(3), sw.Get(0));
    hostDevs.Add(l.Get(0));
    swPorts[0].Add(l.Get(1));
    l = link(fast, hosts.Get(4), sw.Get(0));
    hostDevs.Add(l.Get(0));
    swPorts[0].Add(l.Get(1));
    // S2/S3: 1 <- S1, 2 -> S4
    NetDeviceContainer l24 = link(path, sw.Get(1), sw.Get(3));
    NetDeviceContainer l34 = link(path, sw.Get(2), sw.Get(3));
    swPorts[1].Add(l12.Get(1));
    swPorts[1].Add(l24.Get(0));
    swPorts[2].Add(l13.Get(1));
    swPorts[2].Add(l34.Get(0));
    // S4: 1 S2, 2 S3, 3 edge-1, 4 edge-2, 5 cloud, 6 bg sink
    swPorts[3].Add(l24.Get(1));
    swPorts[3].Add(l34.Get(1));
    l = link(fast, hosts.Get(0), sw.Get(3));
    hostDevs.Add(l.Get(0));
    swPorts[3].Add(l.Get(1));
    l = link(fast, hosts.Get(1), sw.Get(3));
    hostDevs.Add(l.Get(0));
    swPorts[3].Add(l.Get(1));
    l = link(wan, hosts.Get(2), sw.Get(3));
    hostDevs.Add(l.Get(0));
    swPorts[3].Add(l.Get(1));
    l = link(fast, hosts.Get(5), sw.Get(3));
    hostDevs.Add(l.Get(0));
    swPorts[3].Add(l.Get(1));

    // WAN: cloud gateway --(100 Mbps, 40 ms one way)--> cloud data centre
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Mbps")));
    p2p.SetDeviceAttribute("Mtu", UintegerValue(9000));
    p2p.SetChannelAttribute("Delay", TimeValue(MilliSeconds(40)));
    p2p.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("100p"));
    NetDeviceContainer wanDev = p2p.Install(hosts.Get(2), cloud);

    // controller
    Ptr<OFSwitch13InternalHelper> of = CreateObject<OFSwitch13InternalHelper>();
    of->SetDeviceAttribute("CpuCapacity", DataRateValue(DataRate("10Gbps")));
    g_ctrl = CreateObject<SecController>();
    of->InstallController(ctrl, g_ctrl);
    Ptr<OFSwitch13Device> ofd[4];
    for (int i = 0; i < 4; ++i)
    {
        ofd[i] = of->InstallSwitch(sw.Get(i), swPorts[i]);
    }
    g_ctrl->dpS1 = ofd[0]->GetDatapathId();
    g_ctrl->dpS2 = ofd[1]->GetDatapathId();
    g_ctrl->dpS3 = ofd[2]->GetDatapathId();
    g_ctrl->dpS4 = ofd[3]->GetDatapathId();
    of->CreateOpenFlowChannels();
    // OFSwitch13 turns on checksums; this model never rewrites packet fields, so CRC/checksum
    // computation is only simulation overhead -> disable it.
    GlobalValue::Bind("ChecksumEnabled", BooleanValue(false));

    // ---------------- IP ----------------
    InternetStackHelper inet;
    inet.Install(uavs);
    inet.Install(mw);
    inet.Install(hosts);
    inet.Install(cloud);
    Ipv4AddressHelper ip;
    ip.SetBase("10.2.0.0", "255.255.0.0");
    Ipv4InterfaceContainer apIf = ip.Assign(apDev);
    ip.Assign(staDev);
    // wired OpenFlow domain: order of hostDevs = mw, attacker, bg src, edge-1, edge-2, cloud, bg sink
    Ipv4InterfaceContainer wired;
    const char* addr[7] = {"10.1.1.1", "10.1.1.5", "10.1.1.6", "10.1.1.2",
                           "10.1.1.3", "10.1.1.4", "10.1.1.7"};
    for (int i = 0; i < 7; ++i)
    {
        Ptr<NetDevice> d = hostDevs.Get(i);
        Ptr<Ipv4> ipv4 = d->GetNode()->GetObject<Ipv4>();
        int32_t ifi = ipv4->AddInterface(d);
        ipv4->AddAddress(ifi, Ipv4InterfaceAddress(Ipv4Address(addr[i]), Ipv4Mask("255.255.255.0")));
        ipv4->SetUp(ifi);
    }
    {
        Ipv4AddressHelper w;
        w.SetBase("10.1.2.0", "255.255.255.0");
        w.Assign(wanDev); // gateway 10.1.2.1, cloud 10.1.2.2
        Ipv4StaticRoutingHelper sr;
        Ptr<Ipv4> mwIp = mw->GetObject<Ipv4>();
        sr.GetStaticRouting(mwIp)->AddNetworkRouteTo(
            Ipv4Address("10.1.2.0"), Ipv4Mask("255.255.255.0"), Ipv4Address("10.1.1.4"),
            mwIp->GetInterfaceForDevice(hostDevs.Get(0)));
        sr.GetStaticRouting(cloud->GetObject<Ipv4>())->SetDefaultRoute(Ipv4Address("10.1.2.1"), 1);
    }
    for (int i = 0; i < 7; ++i)
    {
        Ptr<NetDevice> di = hostDevs.Get(i);
        Ptr<Ipv4L3Protocol> l3 = di->GetNode()->GetObject<Ipv4L3Protocol>();
        Ptr<Ipv4Interface> iface = l3->GetInterface(l3->GetInterfaceForDevice(di));
        Ptr<ArpCache> cache = iface->GetArpCache();
        for (int j = 0; j < 7; ++j)
        {
            if (i == j)
                continue;
            ArpCache::Entry* e = cache->Add(Ipv4Address(addr[j]));
            e->SetMacAddress(hostDevs.Get(j)->GetAddress());
            e->MarkPermanent();
        }
    }

    // ---------------- applications ----------------
    for (int s = 2; s <= 4; ++s)
    {
        g_srv[s] = CreateObject<ServerApp>();
        g_srv[s]->site = s;
        (s == 4 ? cloud : hosts.Get(s - 2))->AddApplication(g_srv[s]);
        g_srv[s]->SetStartTime(Seconds(0.1));
    }
    g_mw = CreateObject<MiddlewareApp>();
    g_mw->srvAddr[2] = Ipv4Address("10.1.1.2");
    g_mw->srvAddr[3] = Ipv4Address("10.1.1.3");
    g_mw->srvAddr[4] = Ipv4Address("10.1.2.2");
    g_mw->uw.resize(g.nUav);
    mw->AddApplication(g_mw);
    g_mw->SetStartTime(Seconds(0.1));
    for (uint32_t i = 0; i < g.nUav; ++i)
    {
        Ptr<UavApp> a = CreateObject<UavApp>();
        a->idx = i;
        a->malicious = (i < 2); // two compromised UAVs inject fake tasks when the attack is on
        a->mwAddr = apIf.GetAddress(0);
        uavs.Get(i)->AddApplication(a);
        a->SetStartTime(Seconds(0.5 + 0.01 * i));
    }
    for (int k = 0; k < 2; ++k)
    {
        Ptr<UdpFlood> b = CreateObject<UdpFlood>();
        b->dst = Ipv4Address("10.1.1.7");
        b->dstPort = 9001 + k;
        b->background = true;
        b->pktSize = 8000;
        b->rateBps = U->GetValue(15e6, 35e6);
        hosts.Get(4)->AddApplication(b);
        b->SetStartTime(Seconds(0.2));
        g_bg.push_back(b);
    }
    Ptr<UdpFlood> atk = CreateObject<UdpFlood>();
    atk->dst = Ipv4Address("10.1.1.7");
    atk->dstPort = 9999;
    atk->rateBps = g.ddosMbps * 1e6;
    atk->pktSize = 4000;
    atk->nSrcPorts = 40;
    hosts.Get(3)->AddApplication(atk);
    atk->SetStartTime(Seconds(0.2));
    PacketSinkHelper sink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), 9001));
    sink.Install(hosts.Get(5));
    sink.SetAttribute("Local", AddressValue(InetSocketAddress(Ipv4Address::GetAny(), 9002)));
    sink.Install(hosts.Get(5));
    sink.SetAttribute("Local", AddressValue(InetSocketAddress(Ipv4Address::GetAny(), 9999)));
    sink.Install(hosts.Get(5));

    // switch port statistics for the two paths (S2 / S3 ingress from S1)
    l12.Get(1)->TraceConnectWithoutContext("PhyRxEnd", MakeBoundCallback(&PathRx, 0));
    l13.Get(1)->TraceConnectWithoutContext("PhyRxEnd", MakeBoundCallback(&PathRx, 1));
    Simulator::Schedule(Seconds(0.1), &UtilSample);
    Simulator::Schedule(Seconds(1.0), &BackgroundWalk);
    Simulator::Schedule(Seconds(1.0), &DetectorTick);

    // ---------------- run ----------------
    double stop;
    if (g.mode == "train" || g.mode == "collect")
    {
        if (g.mode == "train")
        {
            g_trainLog.open(g.out + "/train_" + g.method + "_s" + std::to_string(g.seed) + ".csv");
            g_trainLog << "window,scenario,reward,tasks,rejection,latency,eps,loss\n";
        }
        else
        {
            g_detCsv.open(g.out + (g.seed == 1 ? "/det_train.csv" : "/det_test.csv"));
        }
        Simulator::Schedule(Seconds(1.0), &WindowBoundary);
        stop = 1.0 + g.windows * g.window;
    }
    else
    {
        Simulator::Schedule(Seconds(1.0), &SetScenario, g.scenario, g.warmup);
        stop = g.simTime;
    }
    auto w0 = std::chrono::steady_clock::now();
    Simulator::Stop(Seconds(stop));
    Simulator::Run();
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();

    if (g.mode == "train")
    {
        g_agent->Save(modelFile);
        std::cout << "saved model, wall time " << wall << " s" << std::endl;
    }
    if (g.mode == "eval")
    {
        Stats& s = g_eval;
        double T = g.simTime - g.warmup;
        double cap = 0;
        int s0 = M.mw ? 1 : 2;
        double work = 0;
        for (int k = s0; k <= 4; ++k)
        {
            cap += CPU_HZ[k] * T;
            work += s.work[k];
        }
        std::string fn = g.out + "/eval.csv";
        bool hdr = !std::ifstream(fn).good();
        std::ofstream os(fn, std::ios::app);
        if (hdr)
        {
            os << "method,n_uav,arrival,scenario,seed,train_seed,latency,energy,cost,rejection,acceptance,"
                  "utilization,throughput,optimal_path,unsafe_offload,decision_us,flowmods_per_s,"
                  "n_tasks,det_ddos_s,det_fake_s,det_comp_s,wall_s,ddos_mbps,drop_prob,mask_th\n";
        }
        os << g.method << ',' << g.nUav << ',' << g.lambda << ',' << g.scenario << ',' << g.seed << ','
           << trainSeed
           << ',' << (s.nDone ? s.latSum / s.nDone : 0) << ',' << s.eSum / std::max(1.0, s.n)
           << ',' << s.costSum / std::max(1.0, s.n) << ',' << s.rej / std::max(1.0, s.n) << ','
           << 1 - s.rej / std::max(1.0, s.n) << ',' << std::min(1.0, work / cap) << ','
           << s.nDone / T << ',' << (s.offl ? s.opt / s.offl : 0) << ','
           << (s.offl ? s.unsafe / s.offl : 0) << ',' << s.decUs / std::max(1.0, s.decN) << ','
           << s.flowMods / g.simTime << ',' << s.n << ',' << g_firstAlarm[0] << ','
           << g_firstAlarm[1] << ',' << g_firstAlarm[2] << ',' << wall << ',' << g.ddosMbps << ','
           << g.dropProb << ',' << TRUST_MASK << '\n';
        std::cout << g.method << " N=" << g.nUav << " lambda=" << g.lambda << " " << g.scenario
                  << " seed=" << g.seed << " tasks=" << s.n
                  << " lat=" << (s.nDone ? s.latSum / s.nDone : 0)
                  << " rej=" << s.rej / std::max(1.0, s.n) << " wall=" << wall << "s" << std::endl;
    }
    if (g.mode != "eval")
    {
        Stats& s = g_eval;
        std::cout << g.mode << " " << g.method << ": legit tasks=" << s.n << " completed=" << s.nDone
                  << " rejection=" << s.rej / std::max(1.0, s.n)
                  << " latency=" << (s.nDone ? s.latSum / s.nDone : 0)
                  << " offloaded=" << s.offl << " flowmods=" << s.flowMods << " wall=" << wall
                  << " s" << std::endl;
        for (auto& kv : g_reason)
            std::cout << "  reject reason " << kv.first << ": " << kv.second << std::endl;
    }
    Simulator::Destroy();
    return 0;
}
