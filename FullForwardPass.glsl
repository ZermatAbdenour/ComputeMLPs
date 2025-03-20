#version 430

// Each workgroup has 10 invocations (one per hidden neuron)
layout(local_size_x = 10) in;  
shared float hidden[10];      

layout(std430, binding = 0) buffer InputBuffer { 
    float input_data[]; 
};
layout(std430, binding = 1) buffer W1Buffer { 
    float w1[]; 
};
layout(std430, binding = 2) buffer B1Buffer { 
    float b1[]; 
};
layout(std430, binding = 3) buffer W2Buffer { 
    float w2[]; 
};
layout(std430, binding = 4) buffer B2Buffer { 
    float b2[]; 
};
layout(std430, binding = 5) buffer OutputBuffer { 
    float outputs[]; 
};

uniform int num_inputs = 784;    // Each sample has 784 features
uniform int output_size = 10;      // Each sample produces 10 outputs

void main() {
    // 'neuron_idx' within the current sample (0 to 9)
    uint neuron_idx = gl_LocalInvocationID.x;
    // 'sample_idx' identifies which sample this workgroup is processing
    uint sample_idx = gl_WorkGroupID.y;
    
    // Compute the starting indices for the input and output for this sample.
    uint input_offset = sample_idx * num_inputs;
    uint output_offset = sample_idx * output_size;
    
    // --- Hidden Layer Forward Pass ---
    float sum = 0.0;
    for (int i = 0; i < num_inputs; i++) {
        // Weight layout: w1 is stored as a 2D array [num_inputs x 10]
        sum += input_data[input_offset + i] * w1[i * output_size + neuron_idx];  
    }
    
    // Apply ReLU activation
    hidden[neuron_idx] = max(sum + b1[neuron_idx], 0.0);
    
    // Ensure all invocations in the workgroup have computed their hidden value
    barrier();
    
    // --- Output Layer Forward Pass ---
    sum = 0.0;
    for (int i = 0; i < output_size; i++) {
        // Weight layout: w2 is stored as a 2D array [10 x 10]
        sum += hidden[i] * w2[neuron_idx + i * output_size];
    }
    outputs[output_offset + neuron_idx] = sum + b2[neuron_idx];
}
