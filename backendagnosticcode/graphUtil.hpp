#ifndef GRAPHUTILDONE
#define GRAPHUTILDONE

#include<vector>
#include<queue>
#include<stdlib.h>

namespace Kate {

//simple implementation matrix representation only for positive weight
class Graph{
public:
    int n;
    double* weight;
    Graph(int n){
        weight = (double*)malloc(sizeof(double)*n*n);
        this->n = n;
        for (int i = 0; i < n*n; i++) weight[i] = -1;
    }
    double operator()(int i, int j){
        return weight[i*n +j];
    }
    void operator()(int i, int j, double a){
        weight[i*n+j] = a;
    }
    ~Graph(){
        free(weight);
    }
    void print(){
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                std::cout << weight[i*n+j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::vector<int> pathFinding(int i, int j){
        std::priority_queue<std::pair<double, int>, std::vector<std::pair<double,int>>, std::greater<std::pair<double, int>>> q;
        std::vector<double> dist(n, 0);
        std::vector<int> pred(n, -1);

        dist[i] = 0;
        pred[i] = -2;
        q.push(std::make_pair(0, i));

        while (!q.empty()){
            int u = q.top().second;
            q.pop();
            if (u == j) break;
            for (int k = 0; k < n; k++){
                double p = weight[u*n+k];
                if (p < 0){
                    continue;
                }
                if (pred[k] == -1){
                    pred[k] = u;
                    dist[k] = dist[u] + p;
                    q.push(std::make_pair(dist[k], k));
                } else if (dist[k] > dist[u] + p){
                    pred[k] = u;
                    dist[k] = dist[u]+p;
                    q.push(std::make_pair(dist[k], k));
                }
            }
        }
        if (pred[j] == -1) return {-1};
        std::vector<int> result1;
        int parc = j;
        while (parc != -2){
            result1.push_back(parc);
            parc = pred[parc];
        }
        //reverse result
        std::vector<int> result2(result1.size());
        for (int i = 0; i < result1.size(); i++){
            result2[i] = result1[result1.size() - 1 - i];
        }
        return result2;
    }
};

}

#endif