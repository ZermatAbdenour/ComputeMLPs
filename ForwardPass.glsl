#version 430

layout(local_size_x = 10) in;  // Single workgroup with 10 threads
shared float hidden[10];       // Shared memory for hidden layer

layout(std430, binding=0) buffer InputBuffer { float input_data[]; };
layout(std430, binding=1) buffer W1Buffer { float w1[]; };
layout(std430, binding=2) buffer B1Buffer { float b1[]; };
layout(std430, binding=3) buffer W2Buffer { float w2[]; };
layout(std430, binding=4) buffer B2Buffer { float b2[]; };
layout(std430, binding=5) buffer OutputBuffer { float outputs[]; };

void main() {
    uint idx = gl_LocalInvocationID.x;

    // Hidden layer
    float sum = 0.0;
    for (int i = 0; i < 784; i++) {
        sum += input_data[i] * w1[i * 10 + idx];  
    }
    
    hidden[idx] = max(sum + b1[idx], 0.0);
    barrier();
    sum = 0.0;
    for (int i = 0; i < 10; i++) {
        sum += hidden[i] * w2[idx + i * 10];
    }
    outputs[idx] = sum + b2[idx];
}
