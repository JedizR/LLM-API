import os
from mistralai import Mistral
import dotenv
import matplotlib.pyplot as plt
import numpy as np

dotenv.load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

def generate_plot_data(fx, x_min, x_max):
    x = np.linspace(float(x_min), float(x_max), 1000)
    if fx == "x":
        return x, x
    elif fx == "x^2":
        return x, x**2
    elif fx == "sin(x)":
        return x, np.sin(x)
    elif fx == "cos(x)":
        return x, np.cos(x)
    
SYSTEM_PROMPT = """Extract the function and interval from user's request. 
Return format: "function_name,x_min,x_max" or "exit" if user wants to end.
RESTRICTION: RETURN IN THE "function_name,x_min,x_max" ONLY. EXCLUDE ANY OTHER WORD
Valid functions: x, x^2, sin(x), cos(x)
Example: "plot sine from -5 to 5" → "sin(x),-5,5" """

def get_plot_info(user_input):
    chat_response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
    )
    return chat_response.choices[0].message.content

while True:
    user_input = input("input function to plot: ")
    result = get_plot_info(user_input)
    print(f"MODEL RESPONSE: {repr(result)}")
    
    if result.lower() == "exit":
        print("end task")
        break
    
    func_name, x_min, x_max = result.strip().split(",")
    x, y = generate_plot_data(func_name, x_min, x_max)
    
    plt.figure()
    plt.plot(x, y)
    plt.grid(True)
    plt.title(f"Plot of y = {func_name}")
    plt.show()