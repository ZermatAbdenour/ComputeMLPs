#version 430

layout(local_size_x = 10) in;
shared float hidden[10];
shared float logits[10];
shared float exp_logits[10];

// Buffer definitions
layout(std430, binding = 0) buffer InputBuffer { float input_data[]; };
layout(std430, binding = 1) buffer W1Buffer { float w1[]; };
layout(std430, binding = 2) buffer B1Buffer { float b1[]; };
layout(std430, binding = 3) buffer W2Buffer { float w2[]; };
layout(std430, binding = 4) buffer B2Buffer { float b2[]; };
layout(std430, binding = 5) buffer OutputBuffer { float outputs[]; };
layout(std430, binding = 6) buffer HiddenBuffer { float hidden_outputs[]; };  // New hidden layer buffer

uniform int num_inputs = 784;
uniform int output_size = 10;

void main() {
    uint neuron_idx = gl_LocalInvocationID.x;
    uint sample_idx = gl_WorkGroupID.y;
    
    uint input_offset = sample_idx * num_inputs;
    uint output_offset = sample_idx * output_size;
    
    // Hidden layer computation
    float sum = 0.0;
    for (int i = 0; i < num_inputs; i++) {
        sum += input_data[input_offset + i] * w1[i * output_size + neuron_idx];
    }
    hidden[neuron_idx] = max(sum + b1[neuron_idx], 0.0);
    
    barrier();

    // Store hidden layer to buffer
    hidden_outputs[output_offset + neuron_idx] = hidden[neuron_idx];  // New storage

    // Output layer computation
    float output_sum = 0.0;
    for (int i = 0; i < output_size; i++) {
        output_sum += hidden[i] * w2[neuron_idx + i * output_size];
    }
    float logit = output_sum + b2[neuron_idx];
    logits[neuron_idx] = logit;
    
    barrier();

    // Softmax computation
    float max_logit = logits[0];
    for (int i = 1; i < output_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    exp_logits[neuron_idx] = exp(logits[neuron_idx] - max_logit);
    barrier();

    float sum_exp = 0.0;
    for (int i = 0; i < output_size; i++) {
        sum_exp += exp_logits[i];
    }

    outputs[output_offset + neuron_idx] = exp_logits[neuron_idx] / sum_exp;
}