// =====================================================================================
//  nn.h  -  Neural networks in plain C++ (no external library)
//    * Mlp          : fully connected network, ReLU hidden layers, Adam optimiser
//    * DqnAgent     : DQN / Double-DQN with replay buffer and target network
//    * Autoencoder  : anomaly detector (reconstruction error) with z-score normalisation
// =====================================================================================
#ifndef SECSDQNEC_NN_H
#define SECSDQNEC_NN_H

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace secnn
{

using Vec = std::vector<double>;

class Mlp
{
  public:
    Mlp() = default;
    bool tanhHidden = false; // tanh hidden units (used by the autoencoder: cannot extrapolate)

    Mlp(const std::vector<int>& sizes, std::mt19937& rng)
        : m_sizes(sizes)
    {
        for (size_t l = 0; l + 1 < sizes.size(); ++l)
        {
            int in = sizes[l], out = sizes[l + 1];
            double lim = std::sqrt(6.0 / (in + out));
            std::uniform_real_distribution<double> u(-lim, lim);
            Vec w(in * out), b(out, 0.0);
            for (auto& x : w)
            {
                x = u(rng);
            }
            m_w.push_back(w);
            m_b.push_back(b);
        }
        m_mw = m_w;
        m_vw = m_w;
        m_mb = m_b;
        m_vb = m_b;
        for (auto* v : {&m_mw, &m_vw})
        {
            for (auto& x : *v)
            {
                std::fill(x.begin(), x.end(), 0.0);
            }
        }
        for (auto* v : {&m_mb, &m_vb})
        {
            for (auto& x : *v)
            {
                std::fill(x.begin(), x.end(), 0.0);
            }
        }
    }

    // forward pass; keeps activations of the LAST call for backward()
    Vec Forward(const Vec& x)
    {
        m_act.assign(1, x);
        Vec a = x;
        for (size_t l = 0; l < m_w.size(); ++l)
        {
            int in = m_sizes[l], out = m_sizes[l + 1];
            Vec z(m_b[l]);
            for (int i = 0; i < in; ++i)
            {
                double ai = a[i];
                if (ai == 0.0)
                {
                    continue; // (exact zeros only: ReLU-inactive units)
                }
                const double* wr = &m_w[l][i * out];
                for (int j = 0; j < out; ++j)
                {
                    z[j] += ai * wr[j];
                }
            }
            if (l + 1 < m_w.size())
            {
                for (auto& v : z)
                {
                    v = tanhHidden ? std::tanh(v) : std::max(0.0, v);
                }
            }
            m_act.push_back(z);
            a = z;
        }
        return a;
    }

    // accumulate gradient for one sample (grad = dLoss/dOutput of last Forward call)
    void Backward(const Vec& gradOut)
    {
        if (m_gw.empty())
        {
            ZeroGrad();
        }
        Vec g = gradOut;
        for (int l = (int)m_w.size() - 1; l >= 0; --l)
        {
            int in = m_sizes[l], out = m_sizes[l + 1];
            const Vec& a = m_act[l];
            for (int j = 0; j < out; ++j)
            {
                m_gb[l][j] += g[j];
            }
            Vec gPrev(in, 0.0);
            for (int i = 0; i < in; ++i)
            {
                const double* wr = &m_w[l][i * out];
                double* gwr = &m_gw[l][i * out];
                double s = 0.0;
                for (int j = 0; j < out; ++j)
                {
                    gwr[j] += a[i] * g[j];
                    s += wr[j] * g[j];
                }
                if (l == 0)
                    gPrev[i] = 0.0;
                else if (tanhHidden)
                    gPrev[i] = s * (1.0 - a[i] * a[i]);
                else
                    gPrev[i] = a[i] > 0.0 ? s : 0.0;
            }
            g = gPrev;
        }
    }

    void ZeroGrad()
    {
        m_gw = m_w;
        m_gb = m_b;
        for (auto& x : m_gw)
        {
            std::fill(x.begin(), x.end(), 0.0);
        }
        for (auto& x : m_gb)
        {
            std::fill(x.begin(), x.end(), 0.0);
        }
    }

    void AdamStep(double lr, double scale)
    {
        m_t++;
        double c1 = 1.0 - std::pow(0.9, m_t), c2 = 1.0 - std::pow(0.999, m_t);
        auto upd = [&](Vec& p, Vec& g, Vec& m, Vec& v) {
            for (size_t k = 0; k < p.size(); ++k)
            {
                double gk = std::clamp(g[k] * scale, -10.0, 10.0);
                m[k] = 0.9 * m[k] + 0.1 * gk;
                v[k] = 0.999 * v[k] + 0.001 * gk * gk;
                p[k] -= lr * (m[k] / c1) / (std::sqrt(v[k] / c2) + 1e-8);
            }
        };
        for (size_t l = 0; l < m_w.size(); ++l)
        {
            upd(m_w[l], m_gw[l], m_mw[l], m_vw[l]);
            upd(m_b[l], m_gb[l], m_mb[l], m_vb[l]);
        }
        ZeroGrad();
    }

    void CopyWeightsFrom(const Mlp& o)
    {
        m_w = o.m_w;
        m_b = o.m_b;
    }

    void Save(std::ostream& os) const
    {
        os << m_sizes.size();
        for (int s : m_sizes)
        {
            os << ' ' << s;
        }
        os << '\n';
        os.precision(10);
        for (size_t l = 0; l < m_w.size(); ++l)
        {
            for (double x : m_w[l])
            {
                os << x << ' ';
            }
            os << '\n';
            for (double x : m_b[l])
            {
                os << x << ' ';
            }
            os << '\n';
        }
    }

    void Load(std::istream& is)
    {
        size_t n;
        is >> n;
        m_sizes.resize(n);
        for (auto& s : m_sizes)
        {
            is >> s;
        }
        m_w.clear();
        m_b.clear();
        for (size_t l = 0; l + 1 < n; ++l)
        {
            Vec w(m_sizes[l] * m_sizes[l + 1]), b(m_sizes[l + 1]);
            for (auto& x : w)
            {
                is >> x;
            }
            for (auto& x : b)
            {
                is >> x;
            }
            m_w.push_back(w);
            m_b.push_back(b);
        }
    }

  private:
    std::vector<int> m_sizes;
    std::vector<Vec> m_w, m_b, m_mw, m_vw, m_mb, m_vb, m_gw, m_gb, m_act;
    long m_t = 0;
};

// ------------------------------------------------------------------------------------
struct Transition
{
    Vec s;
    int a;
    double r;
    Vec s2;
};

class DqnAgent
{
  public:
    DqnAgent(int stateDim, int nActions, bool doubleDqn, unsigned seed)
        : m_rng(seed),
          m_nA(nActions),
          m_double(doubleDqn)
    {
        std::vector<int> sizes{stateDim, 128, 128, nActions};
        m_q = Mlp(sizes, m_rng);
        m_qt = Mlp(sizes, m_rng);
        m_qt.CopyWeightsFrom(m_q);
    }

    double eps = 1.0;
    double gamma = 0.95, lr = 1e-3;
    size_t batch = 64, capacity = 50000, warmup = 500;
    int targetSync = 200;

    int Act(const Vec& s, bool greedy)
    {
        std::uniform_real_distribution<double> u(0, 1);
        if (!greedy && u(m_rng) < eps)
        {
            return std::uniform_int_distribution<int>(0, m_nA - 1)(m_rng);
        }
        Vec q = m_q.Forward(s);
        return int(std::max_element(q.begin(), q.end()) - q.begin());
    }

    Vec QValues(const Vec& s)
    {
        return m_q.Forward(s);
    }

    void Store(const Transition& t)
    {
        if (m_buf.size() < capacity)
        {
            m_buf.push_back(t);
        }
        else
        {
            m_buf[m_pos] = t;
        }
        m_pos = (m_pos + 1) % capacity;
    }

    // one gradient step on a random mini-batch; returns Huber loss
    double Learn()
    {
        if (m_buf.size() < warmup)
        {
            return -1;
        }
        std::uniform_int_distribution<size_t> pick(0, m_buf.size() - 1);
        double loss = 0;
        for (size_t k = 0; k < batch; ++k)
        {
            const Transition& t = m_buf[pick(m_rng)];
            Vec qn = m_qt.Forward(t.s2);
            double next;
            if (m_double)
            { // DDQN: online net selects, target net evaluates
                Vec qo = m_q.Forward(t.s2);
                int a = int(std::max_element(qo.begin(), qo.end()) - qo.begin());
                next = qn[a];
            }
            else
            { // DQN: target net selects and evaluates
                next = *std::max_element(qn.begin(), qn.end());
            }
            double y = t.r + gamma * next;
            Vec q = m_q.Forward(t.s);
            double td = q[t.a] - y;
            loss += std::fabs(td) < 1 ? 0.5 * td * td : std::fabs(td) - 0.5;
            Vec g(m_nA, 0.0);
            g[t.a] = std::clamp(td, -1.0, 1.0);
            m_q.Backward(g);
        }
        m_q.AdamStep(lr, 1.0 / batch);
        if (++m_steps % targetSync == 0)
        {
            m_qt.CopyWeightsFrom(m_q);
        }
        return loss / batch;
    }

    void Save(const std::string& f) const
    {
        std::ofstream os(f);
        m_q.Save(os);
    }

    void Load(const std::string& f)
    {
        std::ifstream is(f);
        m_q.Load(is);
        m_qt.CopyWeightsFrom(m_q);
    }

  private:
    std::mt19937 m_rng;
    int m_nA;
    bool m_double;
    Mlp m_q, m_qt;
    std::vector<Transition> m_buf;
    size_t m_pos = 0;
    long m_steps = 0;
};

// ------------------------------------------------------------------------------------
class Autoencoder
{
  public:
    double threshold = 1e9;

    void Fit(const std::vector<Vec>& X, int bottleneck, int epochs, unsigned seed)
    {
        int d = X[0].size();
        m_mu.assign(d, 0);
        m_sd.assign(d, 0);
        for (auto& x : X)
        {
            for (int i = 0; i < d; ++i)
            {
                m_mu[i] += x[i] / X.size();
            }
        }
        for (auto& x : X)
        {
            for (int i = 0; i < d; ++i)
            {
                m_sd[i] += std::pow(x[i] - m_mu[i], 2) / X.size();
            }
        }
        for (auto& s : m_sd)
        {
            s = std::sqrt(s) + 1e-6;
        }
        std::mt19937 rng(seed);
        m_net = Mlp({d, 16, bottleneck, 16, d}, rng);
        m_net.tanhHidden = true;
        std::vector<size_t> idx(X.size());
        for (size_t i = 0; i < idx.size(); ++i)
        {
            idx[i] = i;
        }
        for (int e = 0; e < epochs; ++e)
        {
            std::shuffle(idx.begin(), idx.end(), rng);
            for (size_t k = 0; k < idx.size(); ++k)
            {
                Vec z = Norm(X[idx[k]]);
                Vec o = m_net.Forward(z);
                Vec g(d);
                for (int i = 0; i < d; ++i)
                {
                    g[i] = 2 * (o[i] - z[i]) / d;
                }
                m_net.Backward(g);
                if ((k + 1) % 32 == 0 || k + 1 == idx.size())
                {
                    m_net.AdamStep(2e-3, 1.0 / 32);
                }
            }
        }
    }

    double Score(const Vec& x)
    {
        Vec z = Norm(x);
        Vec o = m_net.Forward(z);
        double s = 0;
        for (size_t i = 0; i < z.size(); ++i)
        {
            s += (o[i] - z[i]) * (o[i] - z[i]);
        }
        return s / z.size();
    }

    void Calibrate(const std::vector<Vec>& Xval, double pct)
    {
        std::vector<double> s;
        for (auto& x : Xval)
        {
            s.push_back(Score(x));
        }
        std::sort(s.begin(), s.end());
        threshold = s[std::min(s.size() - 1, size_t(pct / 100.0 * s.size()))];
    }

    bool Alarm(const Vec& x)
    {
        return Score(x) > threshold;
    }

    void Save(const std::string& f) const
    {
        std::ofstream os(f);
        os.precision(10);
        os << threshold << '\n' << m_mu.size() << '\n';
        for (double v : m_mu)
        {
            os << v << ' ';
        }
        os << '\n';
        for (double v : m_sd)
        {
            os << v << ' ';
        }
        os << '\n';
        m_net.Save(os);
    }

    bool Load(const std::string& f)
    {
        std::ifstream is(f);
        if (!is)
        {
            return false;
        }
        size_t d;
        is >> threshold >> d;
        m_mu.resize(d);
        m_sd.resize(d);
        for (auto& v : m_mu)
        {
            is >> v;
        }
        for (auto& v : m_sd)
        {
            is >> v;
        }
        m_net.Load(is);
        m_net.tanhHidden = true;
        return true;
    }

  private:
    Vec Norm(const Vec& x) const
    {
        Vec z(x.size());
        for (size_t i = 0; i < x.size(); ++i)
        {
            z[i] = (x[i] - m_mu[i]) / m_sd[i];
        }
        return z;
    }

    Mlp m_net;
    Vec m_mu, m_sd;
};

} // namespace secnn

#endif
