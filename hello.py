import ray
from ray import serve
import requests

# 1. Initialize Ray
ray.init()

# 2. Define the deployment
@serve.deployment
class HelloWorld:
    def __call__(self, request):
        return "Hello world!"

# 3. Deploy the application
# Bind the deployment to arguments (none in this case) and return the application
app = HelloWorld.bind()

# 4. Run the application (for development/testing)
if __name__ == "__main__":
    # serve.run expects an application, which is the result of .bind()
    serve.run(app)
    
    # Test the deployment
    print(requests.get("http://localhost:8000/").text)
