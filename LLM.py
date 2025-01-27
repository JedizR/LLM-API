import os
from mistralai import Mistral
import dotenv
import matplotlib.pyplot as plt
import numpy as np

dotenv.load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

def generate_plot_data(func_info):
   x = np.linspace(float(func_info['x_min']), float(func_info['x_max']), 1000)
   
   if func_info['type'] == 'polynomial':
       y = sum(coef * x**power for power, coef in enumerate(func_info['coeffs'][::-1]))
   else:
       k = func_info['k']
       y = np.sin(k*x) if func_info['func'] == 'sin' else np.cos(k*x)
   return x, y
    
SYSTEM_PROMPT = """Analyze function request and return in JSON format ONLY. No other text.
For polynomials (power 1-4): {"type": "polynomial", "coeffs": [coefficients], "x_min": min, "x_max": max}
For trig: {"type": "trig", "func": "sin/cos", "k": multiplier, "x_min": min, "x_max": max}
for x_min, x_max if the user did not specify, set x_min to -20, set x_max to 20
for trigonometric's k value, if the user did not specify, set k to 1
BE SUPER STRICT WITH THE FORMAT ESPECIALLY THE KEY OF DICTIONARY SHOULD BE SEND EXACTLY
Return {"type": "exit"} for end request.
Examples:
"x^3 - 3x^2 + 5x - 1 from -2 to 2" → {"type": "polynomial", "coeffs": [1,-3,5,-1], "x_min": -2, "x_max": 2}
"x^4 + 2x^3 - x^2 + 10x" → {"type": "polynomial", "coeffs": [1,2,-1,10,0], "x_min": -20, "x_max": 20}
"sin(3x) from 0 to pi" → {"type": "trig", "func": "sin", "k": 3, "x_min": 0, "x_max": 3.14159}"""


def get_plot_info(user_input):
   response = client.chat.complete(
       model=model,
       messages=[
           {"role": "system", "content": SYSTEM_PROMPT},
           {"role": "user", "content": user_input}
       ]
   )
   result = response.choices[0].message.content
   return result if result == "exit" else eval(result)

def get_function_title(result):
   if result['type'] == 'polynomial':
       powers = range(len(result['coeffs'])-1, -1, -1)
       terms = []
       for p, c in zip(powers, result['coeffs']):
           if c == 0:
               continue
           if p == 0:
               terms.append(f"{c}")
           elif p == 1:
               terms.append(f"{c}x")
           else:
               terms.append(f"{c}x^{p}")
       return "y = " + " + ".join(terms).replace("+ -", "- ")
   else:
       return f"y = {result['func']}({result['k']}x)"

while True:
    user_input = input("input function to plot: ")
    result = get_plot_info(user_input)
    print(f"MODEL RESPONSE: {repr(result)}")
    
    if result['type'] == "exit":
        print("end task")
        break
    
    x, y = generate_plot_data(result)
    
    plt.figure()
    plt.plot(x, y)
    plt.grid(True)
    if result['type'] == "polynomial":
        plt.title(f"Plot of polynomial of coefficients: {result["coeffs"]}'")
    plt.title(f"Plot of ")
    plt.show()