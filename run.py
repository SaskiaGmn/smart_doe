from flask import Flask
from flask_app.routes import bp

# Create Flask app with proper template and static folder configuration
app = Flask(__name__, 
           template_folder='flask_app/templates', 
           static_folder='flask_app/static')

# Register the blueprint
app.register_blueprint(bp)

if __name__ == "__main__":
    app.run(debug=True)