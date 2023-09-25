import torch
import torch.nn as nn
import torch.nn.functional as F


class Information_Processing(nn.Module):
    """
    A module for processing information using an LSTM.

    Args:
        input_size (int): The input size.
        hidden_size (int): The size of the hidden state (default is 16).
        num_layers (int): The number of LSTM layers (default is 2).
    """
    def __init__(self, input_size: int, hidden_size: int = 16, num_layers: int = 2):
        super(Information_Processing, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        """
        Forward pass of the Information_Processing module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out, _ = self.lstm(x)
        return out


class Action_Generation(nn.Module):
    """
    A module for action generation using an LSTM.

    Args:
        input_size (int): The input size.
        hidden_size (int): The size of the hidden state.
        output_size (int): The output size.
        num_layers (int): The number of LSTM layers.
        num_actions (int): The number of actions.
    """
    def __init__(self, input_size, hidden_size: int, output_size, num_actions, num_layers: int = 2):
        super(Action_Generation, self).__init__()
        self.num_actions = num_actions
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, original_input):
        """
        Forward pass of the Action_Generation module.

        Args:
            information_output (torch.Tensor): Information processing output.
            original_input (torch.Tensor): Original input.

        Returns:
            torch.Tensor: Action matrix.
        """
        # Concatenate the two input matrices along a specified dimension (e.g., dimension 2)
        combined_input = original_input
        # combined_input = torch.cat((information_output, original_input), dim=2)
        lstm_out, _ = self.lstm(combined_input)
        action_matrix = self.fc(lstm_out)
        return action_matrix[:, :self.num_actions, :]


class Action_Probability(nn.Module):
    """
    A module for calculating action probabilities based on action generation and information processing outputs.

    Args:
        action_generation_output_size (int): The size of the action generation output.
        information_processing_output_size (int): The size of the information processing output.
        hidden_size (int): The size of the hidden layer in the DNN.

    """
    def __init__(self, action_generation_output_size, num_actions, information_processing_output_size, information_seq_size, hidden_size):
        super(Action_Probability, self).__init__()

        # Create DNN structure for action generation output
        self.dnn_action_generation = nn.Sequential(
            nn.Linear(action_generation_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_generation_output_size),
            nn.Softmax(dim=-1)
        )

        # Create DNN structure for information processing output
        self.dnn_information_processing = nn.Sequential(
            nn.Linear(information_processing_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_generation_output_size),
            nn.Conv1d(information_seq_size, num_actions, 1),
            nn.Softmax(dim=-1)
        )

    def forward(self, action_generation_output, information_processing_output):
        """
        Forward pass of the Action_Probability module.

        Args:
            action_generation_output (torch.Tensor): Action generation output.
            information_processing_output (torch.Tensor): Information processing output.

        Returns:
            torch.Tensor: Normalized action probabilities.
        """
        # Pass action generation output through its DNN
        action_generation_probabilities = self.dnn_action_generation(action_generation_output)

        # Pass information processing output through its DNN
        information_processing_probabilities = self.dnn_information_processing(information_processing_output)

        # Combine the probabilities (e.g., take an element-wise product)
        combined_probabilities = action_generation_probabilities * information_processing_probabilities

        # Normalize the combined probabilities to ensure the sum is 1 along the num_actions dimension
        normalized_probabilities = combined_probabilities / combined_probabilities.sum(dim=-1, keepdim=True)

        return normalized_probabilities


class Action_Output(nn.Module):
    def __init__(self, num_actions):
        super(Action_Output, self).__init__()
        self.num_actions = num_actions

    def forward(self, action_generation_output, action_probability_output):
        """
        Forward pass of the Action_Output module.

        Args:
            action_generation_output (torch.Tensor): Action generation output.
            action_probability_output (torch.Tensor): Action probability output.

        Returns:
            torch.Tensor: Selected actions.
        """
        # Sample actions based on the action probabilities
        batch_size, seq_len, _ = action_probability_output.shape

        _v = action_probability_output.view(seq_len, -1)
        action_indices = torch.multinomial(_v, 1)

        action_indices = action_indices.view(batch_size, -1, 1)

        return action_indices.squeeze(2)


if __name__ == '__main__':
    def IP_DEMO():
        input_size = 8
        hidden_size = 16
        num_layers = 2

        network = Information_Processing(input_size, hidden_size, num_layers)

        # Generate some sample input Nanjing
        m = 5
        n = 8
        input_data = torch.randn(m, 1 , n)

        # Forward pass through the network
        output, _ = network(input_data)

        # Print the shape of the output
        print("Output Shape:", output.shape)
        print(type(output.shape))

    def AG_DEMO():
        # Define the input size and output size for Action_Generation
        input_H_size = 5
        input_V_size = 8
        input_size_info_processing = 16  # Adjust this based on your InformationProcessing network's hidden size

        output_size_action_generation = 5  # Adjust this based on your desired number of choices (Y)
        num_actions = 5

        # Create instances of the networks
        info_processing_net = Information_Processing(input_V_size, input_size_info_processing)

        # Define LSTM parameters for Action_Generation
        hidden_size_action_generation = 32
        num_layers_action_generation = 2

        action_generation_net = Action_Generation(
            input_size=input_size_info_processing + input_V_size,
            hidden_size=hidden_size_action_generation,
            output_size=output_size_action_generation,
            num_layers=num_layers_action_generation,
            num_actions=num_actions
        )

        # Create a sample input matrix (original Nanjing)
        original_input_matrix = torch.randn(1, input_H_size, input_V_size)  # Batch size is 1

        # Pass the input through Information Processing
        information_output = info_processing_net(original_input_matrix)


        # Pass both matrices through Action Generation
        action_matrix = action_generation_net(information_output, original_input_matrix)

        # Print the shape of the action_matrix
        print("Action Matrix Shape:", action_matrix.shape)

    def AP_DEMO():
        # Example usage
        input_H_size = 5
        input_V_size = 8
        input_size_info_processing = 16
        output_size_action_generation = 10
        num_actions = 5

        # Create instances of the networks
        info_processing_net = Information_Processing(input_V_size, input_size_info_processing)

        hidden_size_action_generation = 32
        num_layers_action_generation = 2
        action_generation_net = Action_Generation(
            input_size=input_size_info_processing + input_V_size,
            hidden_size=hidden_size_action_generation,
            output_size=output_size_action_generation,
            num_layers=num_layers_action_generation,
            num_actions=num_actions
        )

        action_probability_net = Action_Probability(
            action_generation_output_size=output_size_action_generation,
            information_processing_output_size=input_size_info_processing,
            hidden_size=64  # Adjust the hidden size as needed
        )

        # Create a sample input matrix (original Nanjing)
        original_input_matrix = torch.randn(1, input_H_size, input_V_size)  # Batch size is 1

        # Pass the input through Information Processing
        information_output = info_processing_net(original_input_matrix)

        # Pass both matrices through Action Generation
        action_matrix = action_generation_net(information_output, original_input_matrix)

        # Pass both matrices through Action Probability
        action_probabilities = action_probability_net(action_matrix, information_output)

        # Print the shape of the action_probabilities
        print("Action Generation Shape:", action_matrix.shape)
        print("Action Probabilities Shape:", action_probabilities.shape)

    def AO_DEMO():
        # Example usage
        input_H_size = 5
        input_V_size = 8
        input_size_info_processing = 16
        output_size_action_generation = 7
        num_actions = 5

        # Create instances of the networks
        info_processing_net = Information_Processing(input_V_size, input_size_info_processing)

        hidden_size_action_generation = 32
        num_layers_action_generation = 2
        action_generation_net = Action_Generation(
            input_size=input_size_info_processing + input_V_size,
            hidden_size=hidden_size_action_generation,
            output_size=output_size_action_generation,
            num_layers=num_layers_action_generation,
            num_actions=num_actions
        )

        action_probability_net = Action_Probability(
            action_generation_output_size=output_size_action_generation,
            information_processing_output_size=input_size_info_processing,
            hidden_size=64,  # Adjust the hidden size as needed
            num_actions=num_actions,
            information_seq_size=input_H_size

        )

        action_output_net = Action_Output(
            num_actions=num_actions
        )  # Create an instance of Action_Output

        # Create a sample input matrix (original Nanjing)
        original_input_matrix = torch.randn(1, input_H_size, input_V_size)  # Batch size is 1
        # Pass the input through Information Processing
        information_output = info_processing_net(original_input_matrix)
        # Pass both matrices through Action Generation
        action_matrix = action_generation_net(original_input_matrix)
        # Pass both matrices through Action Probability
        action_probabilities = action_probability_net(action_matrix, information_output)
        # Pass Action Generation output and Action Probability output through Action Output
        action_scores = action_output_net(action_matrix, action_probabilities)

        # Print the action scores
        print("input: ", original_input_matrix)
        print("in: ", original_input_matrix.shape)
        print("IP:", information_output.shape)
        print("AG:", action_matrix.shape)
        print("AP:", action_probabilities.shape)
        print("AO:", action_scores.shape)

    AO_DEMO()
