#pragma once

struct Body {
    float x, y;
    float vx, vy;
    float mass;
};

struct BoundingBox {
    float x, y, width, height;
};

struct Node {
    BoundingBox bounds;

    float totalMass;
    float centerMassX;
    float centerMassY;

    union {
        /*
        0 | 1
        --|--
        2 | 3   
        */
        int children[4]; 
        int bodyIndex;    
    };
    bool isLeaf;
};


extern "C" void runSimulationStep(Body * dev_bodies, float dt, int N);