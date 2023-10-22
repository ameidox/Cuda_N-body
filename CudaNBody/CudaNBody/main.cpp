#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <SFML/Graphics.hpp>
#include <SFML/Window/Mouse.hpp>
#include "main.h"


int main() {

	sf::RenderWindow window(sf::VideoMode(1000, 1000), "N-Body Simulation");
	window.setFramerateLimit(144);

    const int N = 1000;
    const float dt = 0.01f;
    Body* bodies = new Body[N];
    Body* dev_bodies = nullptr;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(480.0, 620.0); 


    for (int i = 0; i < N; i++) {
        bodies[i].x = dis(gen);
        bodies[i].y = dis(gen);
        bodies[i].vx = 0; 
        bodies[i].vy = 0;
        bodies[i].mass = 1.0f;
    }

    cudaMalloc((void**)&dev_bodies, N * sizeof(Body));
    cudaMemcpy(dev_bodies, bodies, N * sizeof(Body), cudaMemcpyHostToDevice);
    const int MAX_NODES = 1000000;  // example size
    Node* d_nodePool;  // The node pool on the GPU
    int* d_nodePoolCounter;  // The counter on the GPU

    cudaMalloc(&d_nodePool, MAX_NODES * sizeof(Node));
    cudaMalloc(&d_nodePoolCounter, sizeof(int));
    cudaMemset(d_nodePoolCounter, 0, sizeof(int));

    sf::VertexArray particlesVertexArray(sf::Points, N);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // Run the simulation step
        runSimulationStep(dev_bodies, dt, N);

        // Retrieve updated positions
        cudaMemcpy(bodies, dev_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost);

        window.clear();

        sf::Vector2i mousePosition = sf::Mouse::getPosition(window);

        for (int i = 0; i < N; i++) {
            particlesVertexArray[i].position = sf::Vector2f(bodies[i].x, bodies[i].y);
            particlesVertexArray[i].color = sf::Color(255, 255, 255, 76);
        }
        window.draw(particlesVertexArray);
        window.display();
    }

    delete[] bodies;
    cudaFree(d_nodePool);
    cudaFree(d_nodePoolCounter);
    cudaFree(dev_bodies);
    cudaFree(dev_bodies);
    return 0;
}