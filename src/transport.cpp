#pragma GCC optimize("Ofast", "unroll-loops")
#pragma GCC target("avx", "avx2", "bmi", "bmi2", "popcnt", "lzcnt")

#include<bits/stdc++.h>
using namespace std;
#define int long long
#define double long double
#define pii pair<int, int>

// SHARED VARIABLES & UTILS
mt19937 rng(0);
constexpr int INF = 9e18;
constexpr int MAXN = 1e5+5;

int n, width, k;
int source[MAXN][3], target[MAXN][3]; double weights[MAXN];

pii get_coords(int idx, int width) {
    return {idx/width, idx%width};
}

template<typename T>
T sq(T x) { return x*x; }

// SIM ANNEAL
constexpr int NUM_ITER = 5e7;
constexpr double FINAL_TEMP = 1e-4;

struct SimAnneal {
    double temp = 1e-1;
    double factor = 0.9999;
    bool verbose = true;
    int max_it = 1e6;

    int assignment[MAXN];
    int cost = 0;

    SimAnneal(double temp_0, double factor, int max_it, bool verbose = true) : temp(temp_0), factor(factor), max_it(max_it), verbose(verbose) {
        iota(assignment, assignment+n, 0);
        for(int i = 0; i < n; i++)
            cost += get_cost(i, assignment[i]);
    }
    
    bool sa(int val) {
        if(temp >= 1e-30) temp *= factor;
        if(val <= cost) return true;
        else if(temp < 1e-20) return false; // avoid numerical issues
        else return (exp((cost-val)/temp) > uniform_real_distribution<double>(0.0, 1.0)(rng));
    }
    int get_cost(int i, int j) { // move source pixel i to location j
        pii source_coords = get_coords(i, width), target_coords = get_coords(j, width);
        int dist = sq(source_coords.first - target_coords.first) + sq(source_coords.second - target_coords.second);
        int col = sq(source[i][0] - target[j][0]) + sq(source[i][1] - target[j][1]) + sq(source[i][2] - target[j][2]);
        return sq(dist) + (int)(col * weights[j]);
    }
    void step(int it) {
        if(verbose && it % 1000000 == 0) 
            cerr << "[sim anneal] it: " << it << ", temp: " << temp << ", cost: " << cost << "\n";
        pii choice = {uniform_int_distribution<int>(0, n-1)(rng), uniform_int_distribution<int>(0, n-1)(rng)};
        int i = choice.first, j = choice.second;
        if(i == j) return;
        int new_cost = cost - get_cost(i, assignment[i]) - get_cost(j, assignment[j]) + get_cost(i, assignment[j]) + get_cost(j, assignment[i]);
        if(sa(new_cost)) {
            swap(assignment[i], assignment[j]);
            cost = new_cost;
        }
    }
};

signed main() {
    freopen("../data/transport_input.txt", "r", stdin);
    freopen("../data/transport_output.txt", "w", stdout);
    cin.tie(0)->sync_with_stdio(0);
    cout.tie(0)->sync_with_stdio(0);

    cin >> width;
    n = width*width;
    for(int i = 0; i < n; i++) cin >> target[i][0] >> target[i][1] >> target[i][2];
    for(int i = 0; i < n; i++) cin >> source[i][0] >> source[i][1] >> source[i][2];
    for(int i = 0; i < n; i++) cin >> weights[i];

    SimAnneal sa(1e12, pow(FINAL_TEMP/1e12,1./(double)NUM_ITER), NUM_ITER, true);
    for(int it = 1; it <= sa.max_it; it++) sa.step(it);

    for(int i = 0; i < n; i++) cout << sa.assignment[i] << " ";
    cout << "\n";
} 