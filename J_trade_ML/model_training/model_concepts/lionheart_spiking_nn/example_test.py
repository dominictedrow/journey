import numpy as np
 
""" 
this is test code to demonstrate how a 4D shapes:

None is batch_size = 1, 2 is seq_len per sample, 8 is number of features for each sequence in each sample, 
and 2 is 2 sequences for each feature for each sequence within each sample due to spike train encoding.

The first layer recieves a 4D shape and converts it's into 2D sequences for processing and output, and all other layers
thereafter recieve a 2D shape of sequences and outputs the same 2D shape of sequences, unittil the final error.

"""
def generate_random_weights(input_features, output_units):
    # Ignore this for learning from
    return np.random.rand(input_features, output_units)

""" 
This is really important regarding how the shapes are prepaird. It will be similar to how CustomDenseLayer needs to work.
But for SpikingNeuronLayer, it will be the same, but rather than connection weights, it does accumilation at every unit
until a threshold is reached, which causes the unit to fire a 1 or a 0. It it's not reached, then it always fires a 0 for
for incoming units for each spike train sequence and seq_len of the spike train. 

"""
def fully_connected_operation(input_seq, weights, layer_num, seq_len, spike_train):
    flattened_input = np.reshape(input_seq, (input_seq.shape[0], -1))  # Flatten the input
    print(f"Layer {layer_num}, Seq_len {seq_len}, Spike_train {spike_train}, Flattened Input shape: {flattened_input.shape}")
    output = np.dot(flattened_input, weights)  # Perform matrix multiplication with the weights
    print(f"Layer {layer_num}, Seq_len {seq_len}, Spike_train {spike_train}, Output shape: {output.shape}")
    return output

def process_spike_trains(spike_trains, first_layer_units, second_layer_units, epochs):
    final_accumulator = np.array([0.0])  # This has shape (1,)

    # This is a little related, but really we need to make how this code handels data through the network, but not how processes it.
    features = spike_trains.shape[2]  # Number of features per spike train
    weights_1 = generate_random_weights(features, first_layer_units)
    weights_2 = generate_random_weights(first_layer_units, second_layer_units)
    final_weights = generate_random_weights(second_layer_units, 1)

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        """ 
        This is another really important part regarding how data needs to flows through our CustomDenseLayer and SpikingNeuronLayer. 
        Look at how this works:

        for spike_train in range(spike_trains.shape[3]):  # sequences
                sequence_data = spike_trains[0, seq_len, :, spike_train].reshape(1, -1)
        """
        for seq_len in range(spike_trains.shape[1]):  # sequence_length
            for spike_train in range(spike_trains.shape[3]):  # sequences
                sequence_data = spike_trains[0, seq_len, :, spike_train].reshape(1, -1)

                # Process through the layers (layer1_output is like the first CustomDenseLayer, and layer2_output is like SpikingNeuronLayer and CustomDenseLayer, for all layers after the 1st CustomDenseLayer)
                layer1_output = fully_connected_operation(sequence_data, weights_1, 1, seq_len+1, spike_train+1)
                layer2_output = fully_connected_operation(layer1_output, weights_2, 2, seq_len+1, spike_train+1)
                final_output = fully_connected_operation(layer2_output, final_weights, 3, seq_len+1, spike_train+1)
                
                # Accumulate the outputs
                # Ensure compatibility in dimensions when adding
                final_accumulator += final_output.flatten()  # Convert final_output to a 1D array before adding

    # Ignore this for learning from
    print(f"\nFinal accumulated output: {final_accumulator[0]}")
    return final_accumulator[0]

""" 
This is how data pass through, and how we need our code to work internally for CustomDenseLayer and SpikingNeuronLayer for training. 

Remember this is the starting shape: (1, 2, 8, 2). See how in this code the Flattened Input shape: (1, 8), 8 being the features is just
Spike_train 1 of a total of 2, of Seq_len 1 of 2, and it goes through each layer. Well, that's what we need for our data doing when going
through CustomDenseLayer and SpikingNeuronLayer.

--- Epoch 1/1 ---
Layer 1, Seq_len 1, Spike_train 1, Flattened Input shape: (1, 8)
Layer 1, Seq_len 1, Spike_train 1, Output shape: (1, 10)
Layer 2, Seq_len 1, Spike_train 1, Flattened Input shape: (1, 10)
Layer 2, Seq_len 1, Spike_train 1, Output shape: (1, 10)
Layer 3, Seq_len 1, Spike_train 1, Flattened Input shape: (1, 10)
Layer 3, Seq_len 1, Spike_train 1, Output shape: (1, 1)
Layer 1, Seq_len 1, Spike_train 2, Flattened Input shape: (1, 8)
Layer 1, Seq_len 1, Spike_train 2, Output shape: (1, 10)
Layer 2, Seq_len 1, Spike_train 2, Flattened Input shape: (1, 10)
Layer 2, Seq_len 1, Spike_train 2, Output shape: (1, 10)
Layer 3, Seq_len 1, Spike_train 2, Flattened Input shape: (1, 10)
Layer 3, Seq_len 1, Spike_train 2, Output shape: (1, 1)
Layer 1, Seq_len 2, Spike_train 1, Flattened Input shape: (1, 8)
Layer 1, Seq_len 2, Spike_train 1, Output shape: (1, 10)
Layer 2, Seq_len 2, Spike_train 1, Flattened Input shape: (1, 10)
Layer 2, Seq_len 2, Spike_train 1, Output shape: (1, 10)
Layer 3, Seq_len 2, Spike_train 1, Flattened Input shape: (1, 10)
Layer 3, Seq_len 2, Spike_train 1, Output shape: (1, 1)
Layer 1, Seq_len 2, Spike_train 2, Flattened Input shape: (1, 8)
Layer 1, Seq_len 2, Spike_train 2, Output shape: (1, 10)
Layer 2, Seq_len 2, Spike_train 2, Flattened Input shape: (1, 10)
Layer 2, Seq_len 2, Spike_train 2, Output shape: (1, 10)
Layer 3, Seq_len 2, Spike_train 2, Flattened Input shape: (1, 10)
Layer 3, Seq_len 2, Spike_train 2, Output shape: (1, 1)
 """
# Parameters for demonstration
epochs = 1
first_layer_units = 10
second_layer_units = 10

# This shape:[1, 2, 8, 2] is the same 4D shape as the code we need to fix which has a shpae of (None, 2, 5, 100)
spike_trains = np.random.rand(1, 2, 8, 2)

final_accumulated_output = process_spike_trains(spike_trains, first_layer_units, second_layer_units, epochs)

""" 
For SpikingNeuronLayer remember the following:

1. Independent Accumulation: Each sample in the batch has its own membrane potential that accumulates across its sequence of inputs.
2. Statefulness within Each Sample: Within each sample (sequence), the membrane potential is stateful, meaning it maintains its state across the sequence's timesteps. 
3. Resetting State: At the end of each sequence (sample), you reset the membrane potential. This reset is crucial for starting the processing of a new sequence with a clean state.

 """