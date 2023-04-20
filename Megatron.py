import math

import openai
import requests
import sympy as sp
from aiogram import Bot, Dispatcher, executor, types
from aiogram.utils.markdown import link
import numpy as np


class MegatronNLU: 
    def display(self, text:str=None, image:any=None):
        pass

    def enable_gpt(self, gpt_api_key:str, dict:dict) -> None:
        """
        This method enables you to 
        connect GPT in your software

        :gpt_api_key - your unique OpenAI key
        :dict - dictionary containing the context you provide to OpenAi for GPT customisation
        """

        self._gpt_api_key = gpt_api_key
        self._dict = dict

    def enable_dalle(self, dalle_api_key:str) -> None:
        """
        This method enables you to 
        connect DALL-E in your software

        :gpt_api_key - your unique OpenAI key
        """

        self._dalle_api_key = dalle_api_key

    def generate_respose_gpt(self, text: str):
        """
        Returns the result of text processing 
        with gpt-3.5-turbo

        :text - the text to be answered by the model
        """

        extended_dict = self._gpt_dict
        extended_dict.append({'role': 'user', 'content': text})
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=extended_dict
        )
        return response['choices'][0]['message']['content']
    
    def generate_respose_alpha(text: str, save_as:str=None):
        """
        Returns the name of the picture saved on 
        your PC (standard name is 'output.jpg')

        :text - the text to be answered by the model
        :save_as - (optional) the name of the saved picture
        """

        response = openai.Image.create(
            prompt=text,
            n=1,
            size="1024x1024",
            model="image-alpha-001"
        )

        img_url = response["data"][0]["url"]
        response = requests.get(img_url, stream=True)
        response.raw.decode_content = True
        if save_as is None:
            with open('output.jpg', 'wb') as f:
                f.write(response.content)
                f.close()
                return 'output.jpg'
        else:
            with open(save_as, 'wb') as f:
                f.write(response.content)
                f.close()
                return save_as


class MegaMath(MegatronNLU):
    def __init__(self) -> None:
        super().__init__()

    def derrivate(self, str_functions:str) -> list:
        x = sp.Symbol('x')
        str_functions = [sp.sympify(func) for func in str_functions.split(',')]
        try:
            derr = [sp.diff(func, x) for func in str_functions]
            return derr
        except:
            return 'Error: check the data entered'
    
    def calculate(self, expr:str) -> str:
        result = sp.sympify(expr)
        return result.evalf()
    
    def function_value_in_point(self, str_func:str, str_values:str) -> list:
        x = sp.Symbol('x')
        func = sp.sympify(str_func)
        try:
            results = [func.subs(x, float(val)) for val in str_values.replace(',', ' ').split()]
        except:
            return 'Error: check the definition area'
        return results
    
    def lead_to_the_scientific_species(self, str_numbers:str=None, list_numbers:list=None) -> list:
        if list_numbers is None:
            numbers = [float(number) for number in str_numbers.replace(',', ' ').split()]
        elif str_numbers is None:
            numbers = [float(number) for number in list_numbers]

        optimized = []
        for number in numbers:
            if abs(number == 0):
                optimized.append('0.00')
            elif abs(number) != 0 and abs(number) > 100 or abs(number) < 1:
                exp = int(math.log10(abs(number)))
                coefficient = number/float(10**exp)
                optimized.append('{:.2f}*10^{}'.format(coefficient, exp))
            else:
                optimized.append('{:.2f}'.format(number))
        return optimized
    
    def calculate_integral(self, function:str, a:float, b:float) -> str:
        if a >= b:
            return f"Error: The upper value is greater than the lower value:\n\t({a}, {b}): ({function})dx"
        try:
            x = sp.symbols('x')
            f = sp.sympify(function)
            I = sp.integrate(f, (x, a, b))
            return str(I)
        except:
            return 'Error: check the data entered'
    
    def uncertain_integral(self, functions:str) -> str:
        x = sp.Symbol('x')
        try:
            integrals = [f'{sp.integrate(func, x)} + C' for func in functions.split(',')]
            return str(integrals)
        except:
            return 'Error: check the data entered'
        
    def std(self, str_numbers:str=None, list_numbers:list=None) -> str:
        if list_numbers is None:
            numbers = [float(number) for number in str_numbers.split()]
            return str(np.std(numbers))
        elif str_numbers is None:
            return str(np.std(list_numbers))
    
    def mean(self, str_numbers:str=None, list_numbers:list=None) -> str:
        if list_numbers is None:
            str_numbers = [float(number) for number in str_numbers.split()]
            return str(sum(str_numbers) / len(str_numbers))
        elif str_numbers is None:
            return str(sum(list_numbers) / len(list_numbers))


class MegaPlot(MegaMath):
    def __init__(self) -> None:
        super().__init__()

'''m = MegaMath()
print(m.function_value_in_point('x**2', '1, 2, 3, 4'))
print(m.derrivate('x**2, x + 1'))
print(m.lead_to_the_scientific_species(str_numbers='1 0.000000023, 1000000, -0.1 -11 0'))
print(m.calculate_integral('x**2', 1, 0))
print(m.uncertain_integral('x**2 + x, 2*x'))
print(m.lead_to_the_scientific_species(m.std(str_numbers='1 2 3 4 5 6 0.0001')))
print(m.lead_to_the_scientific_species(m.mean(str_numbers='1 2 3 4 5 6 7 8 0.001 -10000 -5')))
print(m.lead_to_the_scientific_species(list_numbers=[1, 2, 3, 4, 5, 1000, 0.0001, 1]))'''
