import torch
'''
The input-vector is organized in a torch.Size([n, d]) fashion:
-n: number of datapoints
-d: number of dimensions.

-> so n datapoints where each has d dimensions

Therefore:
dataset_xsinx.func_create_dataset(xsinx.function_xsinx, num_datapoints=5, sampling_method="random", noise_level=0, x1_range=(0,6), x2_range=(5,10), x3_range=(100,200)) results in:
tensor([[  1.3362,   7.6596, 130.9058],
        [  4.9034,   9.9346, 112.4839],
        [  5.8313,   8.9086, 154.5993],
        [  3.8676,   5.4702, 156.1881],
        [  4.5060,   6.5246, 162.0561]])
torch.Size([5, 3])

This also holds true for the output, it needs to be in the shape of ([n, d])

'''

# TODO: Delete not used functions

class FunctionFactory:
    @staticmethod
    def check_dimensions(inputs, expected_shape):
        """
        Checks if the input tensor matches the expected shape.
        Raises an exception if the shapes do not match.
        """
        if inputs.shape[1] != expected_shape:
            raise ValueError(f"Expected input shape to have {expected_shape} dimensions, but got {inputs.shape[1]}")
    
    @staticmethod
    def function_xsinx(inputs):
        """
        Calculates x * sin(x) for each dimension and sums the results.
        Works with any number of input dimensions.
        """
        # Calculate x * sin(x) for each dimension
        output = inputs * torch.sin(inputs)
        # Sum over all dimensions
        output = output.sum(dim=1, keepdim=True)
        return output, 1
    
    @staticmethod
    def sum_of_sines(inputs):
        """
        Calculates sin(x) for each dimension and sums the results.
        Works with any number of input dimensions.
        """
        result = torch.sin(inputs)
        return result.sum(dim=1, keepdim=True), 1
    
    @staticmethod
    def multi_inputs(inputs):
        """
        Example for a more complex function with multiple inputs.
        Calculates a weighted sum of the squared inputs.
        """
        # Weights for each dimension (can also be passed as parameters)
        weights = torch.ones(inputs.shape[1], device=inputs.device)
        # Square the inputs and weight them
        result = (inputs ** 2) * weights
        # Sum over all dimensions
        return result.sum(dim=1, keepdim=True), 1

    # Multi-output functions
    @staticmethod
    def multi_output_quadratic(inputs, num_outputs=2):
        """
        Creates multiple outputs based on quadratic functions.
        Each output is a different combination of the input dimensions.
        
        Args:
            inputs: Input tensor of shape [n, d]
            num_outputs: Number of outputs to generate (1-5)
            
        Returns:
            outputs: Output tensor of shape [n, num_outputs]
            num_outputs: Number of outputs
        """
        n, d = inputs.shape
        outputs = torch.zeros(n, num_outputs, device=inputs.device)
        
        for i in range(num_outputs):
            if i == 0:
                # First output: sum of squares
                outputs[:, i] = (inputs ** 2).sum(dim=1)
            elif i == 1:
                # Second output: sum of absolute values
                outputs[:, i] = torch.abs(inputs).sum(dim=1)
            elif i == 2:
                # Third output: product of first two dimensions (if available)
                if d >= 2:
                    outputs[:, i] = inputs[:, 0] * inputs[:, 1]
                else:
                    outputs[:, i] = inputs[:, 0] ** 2
            elif i == 3:
                # Fourth output: maximum value
                outputs[:, i] = torch.max(inputs, dim=1)[0]
            elif i == 4:
                # Fifth output: minimum value
                outputs[:, i] = torch.min(inputs, dim=1)[0]
        
        return outputs, num_outputs
    
    @staticmethod
    def multi_output_trigonometric(inputs, num_outputs=2):
        """
        Creates multiple outputs based on trigonometric functions.
        
        Args:
            inputs: Input tensor of shape [n, d]
            num_outputs: Number of outputs to generate (1-5)
            
        Returns:
            outputs: Output tensor of shape [n, num_outputs]
            num_outputs: Number of outputs
        """
        n, d = inputs.shape
        outputs = torch.zeros(n, num_outputs, device=inputs.device)
        
        for i in range(num_outputs):
            if i == 0:
                # First output: sum of sines
                outputs[:, i] = torch.sin(inputs).sum(dim=1)
            elif i == 1:
                # Second output: sum of cosines
                outputs[:, i] = torch.cos(inputs).sum(dim=1)
            elif i == 2:
                # Third output: sum of sin^2
                outputs[:, i] = (torch.sin(inputs) ** 2).sum(dim=1)
            elif i == 3:
                # Fourth output: sum of cos^2
                outputs[:, i] = (torch.cos(inputs) ** 2).sum(dim=1)
            elif i == 4:
                # Fifth output: alternating sin/cos
                outputs[:, i] = torch.sin(inputs[:, 0]) + torch.cos(inputs[:, min(1, d-1)])
        
        return outputs, num_outputs
    
    @staticmethod
    def multi_output_mixed(inputs, num_outputs=2):
        """
        Creates multiple outputs with mixed function types.
        
        Args:
            inputs: Input tensor of shape [n, d]
            num_outputs: Number of outputs to generate (1-5)
            
        Returns:
            outputs: Output tensor of shape [n, num_outputs]
            num_outputs: Number of outputs
        """
        n, d = inputs.shape
        outputs = torch.zeros(n, num_outputs, device=inputs.device)
        
        for i in range(num_outputs):
            if i == 0:
                # First output: linear combination
                outputs[:, i] = inputs.sum(dim=1)
            elif i == 1:
                # Second output: quadratic combination
                outputs[:, i] = (inputs ** 2).sum(dim=1)
            elif i == 2:
                # Third output: exponential
                outputs[:, i] = torch.exp(-inputs.sum(dim=1))
            elif i == 3:
                # Fourth output: logarithmic (with offset to avoid log(0))
                outputs[:, i] = torch.log(1 + torch.abs(inputs).sum(dim=1))
            elif i == 4:
                # Fifth output: sigmoid
                outputs[:, i] = torch.sigmoid(inputs.sum(dim=1))
        
        return outputs, num_outputs