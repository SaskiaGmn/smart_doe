from flask import Blueprint, render_template, request, jsonify
import torch
import traceback
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
    
    try:
        # Get the number of dimensions and bounds from the request
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data received'})
            
        num_dimensions = data.get('num_dimensions', 3)
        bounds = data.get('bounds')
        sampling_method = data.get('sampling_method', 'lhs')
        main_factors = data.get('main_factors')
        acq_func_type = data.get('acq_func_type', 'LogExp_Improvement')
        is_maximization = data.get('is_maximization', True)
        
        # Initialize the model
        gp_model = setup_first_model(
            num_dimensions=num_dimensions, 
            bounds=bounds,
            sampling_method=sampling_method,
            main_factors=main_factors
        )
        gp_optimizer, current_suggestion = setup_optimization_loop(
            gp_model,
            acq_func_type=acq_func_type,
            is_maximization=is_maximization
        )
        
        # Convert the suggestion to a list
        suggestion_list = current_suggestion.squeeze().tolist()
        
        # Get optimization status
        opt_status = gp_optimizer.get_optimization_status()
        
        return jsonify({
            'status': 'success',
            'suggestion': suggestion_list,
            'optimization_status': opt_status
        })
    except Exception as e:
        print(f"Error in initialize: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@bp.route('/submit_observation', methods=['POST'])
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
        
        # Get updated optimization status
        opt_status = gp_optimizer.get_optimization_status()
        
        return jsonify({
            'status': 'success',
            'suggestion': next_suggestion.squeeze().tolist(),
            'optimization_status': opt_status
        })
    except Exception as e:
        print(f"Error in submit_observation: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        })
