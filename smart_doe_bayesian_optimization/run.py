from flask import Flask, render_template, redirect, url_for, request, jsonify
from web_main import setup_first_model, setup_optimization_loop, get_next_optimization_iteration
import traceback

#template folder needs to be defined
app = Flask(__name__, template_folder='flask_app/templates', static_folder='flask_app/static')

#global variable to store the first model
gp_model = None
#global variable to store the optimizer
gp_optimizer = None
#global variable to store the next value
current_suggestion = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/initialize', methods=['POST'])
def initialize():
    global gp_model, gp_optimizer, current_suggestion
    
    try:
        # Get the number of dimensions and bounds from the request
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data received'})
            
        num_dimensions = data.get('num_dimensions', 3)
        bounds = data.get('bounds')
        
        # Initialize the model
        gp_model = setup_first_model(num_dimensions=num_dimensions, bounds=bounds)
        gp_optimizer, current_suggestion = setup_optimization_loop(gp_model)
        
        # Convert the suggestion to a list
        suggestion_list = current_suggestion.squeeze().tolist()
        
        return jsonify({
            'status': 'success',
            'suggestion': suggestion_list
        })
    except Exception as e:
        print(f"Error in initialize: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/submit_observation', methods=['POST'])
def submit_observation():
    global gp_model, gp_optimizer, current_suggestion
    
    try:
        if gp_model is None or gp_optimizer is None or current_suggestion is None:
            return jsonify({'status': 'error', 'message': 'Model not initialized'})
        
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data received'})
            
        observation = float(data.get('observation'))
        
        # Get the next suggestion
        next_suggestion = get_next_optimization_iteration(
            optimizer=gp_optimizer,
            input_value=observation,
            original_x=current_suggestion
        )
        
        # Update the current suggestion
        current_suggestion = next_suggestion
        
        return jsonify({
            'status': 'success',
            'suggestion': next_suggestion.squeeze().tolist()
        })
    except Exception as e:
        print(f"Error in submit_observation: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)