from flask import Blueprint, render_template, request, jsonify
import torch
from web_main import setup_first_model, setup_optimization_loop, get_next_optimization_iteration

bp = Blueprint('main', __name__)

# Global variables for the optimization state
gp_model = None
gp_optimizer = None
current_suggestion = None

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/initialize', methods=['POST'])
def initialize():
    global gp_model, gp_optimizer, current_suggestion
    
    # Get the number of dimensions and sampling method from the request
    data = request.get_json()
    num_dimensions = data.get('num_dimensions', 3)
    sampling_method = data.get('sampling_method', 'lhs')
    main_factors = data.get('main_factors')
    
    # Initialize the model
    gp_model = setup_first_model(num_dimensions=num_dimensions, sampling_method=sampling_method, main_factors=main_factors)
    gp_optimizer, current_suggestion = setup_optimization_loop(gp_model)
    
    # Convert the suggestion to a list
    suggestion_list = current_suggestion.squeeze().tolist()
    
    return jsonify({
        'status': 'success',
        'suggestion': suggestion_list
    })

@bp.route('/submit_observation', methods=['POST'])
def submit_observation():
    global gp_model, gp_optimizer, current_suggestion
    
    if gp_model is None or gp_optimizer is None or current_suggestion is None:
        return jsonify({'status': 'error', 'message': 'Model not initialized'})
    
    data = request.get_json()
    observation = float(data.get('observation'))
    
    try:
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
        return jsonify({
            'status': 'error',
            'message': str(e)
        })
