import asyncio
import json
import math
import os

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import requests
import sympy as sp
from scipy.interpolate import Rbf
from sympy.parsing.sympy_parser import (implicit_multiplication_application,
                                         parse_expr,
                                         standard_transformations)

from aiogram import Bot, Dispatcher, types
import openai



class TeleMath:
    def __init__(self) -> None:
        self.Models()
        self.Mathematica()
        self.mode = None

    def set_mode(self, text: str):
        self.mode = text

    class Models:
        def __init__(self) -> None:
            self._gpt_token = None
            self._alpha_image_token = None
            self._gpt_dict = None
            self._extended_dict = None

        def gpt(self, token: str, dict: dict):
            self._gpt_token = token
            self._gpt_dict = dict
            openai.api_key = self._gpt_token

        def alpha_image(self, token: str):
            self._alpha_image_token = token
            openai.api_key = self._alpha_image_token

        def generate_respose_gpt(self, text: str):
            self._extended_dict = self._gpt_dict
            self._extended_dict.append({'role': 'user', 'content': text})
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=self._extended_dict
            )
            return response['choices'][0]['message']['content']
        
        def generate_respose_alpha(self, text: str):
            response = openai.Image.create(
                prompt=text,
                n=1,
                size="1024x1024",
                model="image-alpha-001"
            )
            img_url = response["data"][0]["url"]
            response = requests.get(img_url, stream=True)
            response.raw.decode_content = True
            with open("output.jpg", 'wb') as f:
                f.write(response.content)
            f.close()
    
    class Mathematica:
        def __init__(self) -> None:
            pass
        
        def _tick_function(self, tick_loc: list) -> list:
            degrees = 53.3 * tick_loc
            return ['%.0f' % degree for degree in degrees]

        def derivate(self, string: str) -> list:
            der = []
            arr_of_functions = string.split(';')
            for func in arr_of_functions:
                x = sp.Symbol('x')
                der.append(sp.diff(func, x))
            return der

        def find_integral(self, string: str) -> list:
            integrals = []
            arr_of_functions = string.split(';')
            for func in arr_of_functions:
                x = sp.Symbol('x')
                integral = sp.integrate(func, x)
                integrals.append(f'{integral} + C')
            return integrals
        
        def plot_math_function(self, string: str) -> str:
            arr_of_functions = string.split(';')

            fig, ax = plt.subplots()

            ax.grid(True)
            ax2 = ax.twiny()

            for func in arr_of_functions:
                x = sp.Symbol('x') 
                y = sp.sympify(func)
                y_lambda = sp.lambdify(x, y, "numpy")
                xs = np.linspace(-10, 10, 500)
                ys = y_lambda(xs)
                ax.plot(xs, ys, label=func)
            ax.legend()

            ax.set_xlabel('x (Для тригонометрических функций - Радианы)')
            ax.set_ylabel('y')

            ax2.set_xlim(ax.get_xlim())
            new_ticks_location = np.arange(-10, 11, 2.5)
            ax2.set_xticks(new_ticks_location)
            ax2.set_xticklabels(self._tick_function(new_ticks_location))
            ax2.set_xlabel('x (Для тригонометрических функций - Градусы)')

            param = len(os.listdir(path = '.'))
            fig_name = f'plot_{param}.png'
            plt.savefig(fig_name)
            plt.close('all')

            return fig_name
        
        def evaluate_expression(self, str_func: str, str_values: str) -> list:
            x = sp.Symbol('x')
            func = sp.sympify(str_func)

            values = [float(val) for val in str_values.split()]

            results = []

            for val in values:
                result = func.subs(x, val)
                try:
                    eval_result = result.evalf()
                    results.append(eval_result)
                except:
                    return "Ошибка: проверьте область определения"

            return results
        
        def optimize_number(self, x: str) -> str: 
            if abs(x) > 1e3:
                exp = int(math.log10(abs(x)))
                coefficient = x/float(10**exp)

                return '{:.2f}*10^{}'.format(coefficient, exp)
            elif abs(x - round(x, 2)) < 1e-4:
                return '{:.2f}'.format(round(x, 2))
            elif abs(x) < 1 and abs(x) > -1:
                exp = int(math.log10(abs(x))) - 1
                coefficient = x/float(10**exp)

                return '{:.2f}*10^{}'.format(coefficient, exp)
            else:
                return '{:.2f}'.format(x)
        
        def calculate_integral(self, func: str, a: float, b: float) -> float:
            if a >= b:
                return "верхнее значение не может быть меньше нижнего"
            x = sp.symbols('x')
            f = sp.sympify(func)
            I = sp.integrate(f, (x, a, b))
            return float(I)
        
        def calculate_std_dev(self, numbers: list) -> float:
            n = len(numbers)
            if n < 2:
                return 0
            mean = sum(numbers) / n
            variance = sum((x - mean) ** 2 for x in numbers) / (n - 1)
            return math.sqrt(variance)
        
        def calculate_mean(self, numbers: list) -> float:
            mean = sum(numbers) / len(numbers)
            return mean

        class Interpolation:
            def __init__(self, x: list = None, y: list = None) -> None:
                self._x = x
                self._y = y
                self._figure_name = None

            def radians_to_degrees(self, x):
                return x * 180 / np.pi

            def degrees_to_radians(self, x):
                return x * np.pi / 180
            
            def interpolate(self, x: list = None, y: list = None) -> None:
                if self._x is None and self._y is None:
                    self._x = [float(k) for k in x]
                    self._y = [float(k) for k in y]
                else:
                    self._x = [float(k) for k in self._x]
                    self._y = [float(k) for k in self._y]
                
                f = Rbf(self._x, self._y)
                x_new = np.linspace(self._x[0], self._x[-1], 1000)
                y_new = f(x_new)

                fig, ax = plt.subplots()
                ax.grid(True)
                ax2 = ax.secondary_xaxis('top', 
                                         functions=(self.radians_to_degrees, 
                                                    self.degrees_to_radians)
                                         )

                ax.plot(x_new, y_new)
                ax.scatter(self._x, self._y)
                ax.set_xlabel('x (Для тригонометрических функций - Радианы)')
                ax2.set_xlabel('x (Для тригонометрических функций - Градусы)')
                ax.set_ylabel('y')

                ax2.set_xlim(ax.get_xlim())

                param = len(os.listdir(path = '.'))
                self._figure_name = f'plot_{param}.png'
                plt.savefig(self._figure_name)
                plt.close('all')

            def get_plot_name(self) -> str:
                if self._figure_name is not None:
                    return self._figure_name
                
            def set_xy(self, x: list, y: list) -> None:
                self._x = x
                self._y = y
        
        class Approximation:
            def __init__(self, x: list = None, y: list = None) -> None:
                self._x = x
                self._y = y
                self._upper_bound = None
                self._lower_bound = None
                self._figure_name = None

            def radians_to_degrees(self, x):
                return x * 180 / np.pi

            def degrees_to_radians(self, x):
                return x * np.pi / 180

            def set_lower_bound(self, lower_bound: float) -> None:
                self._lower_bound = lower_bound

            def set_upper_bound(self, upper_bound: float):
                self._upper_bound = upper_bound
            
            def set_bounds(self, lower_bound: float, upper_bound: float) -> None:
                self._lower_bound = lower_bound
                self._upper_bound = upper_bound

            def set_xy(self, x: list, y: list) -> None:
                self._x = x
                self._y = y

            def approximate(self, x: list = None, y: list = None, lower_bound: float = None, upper_bound: float = None) -> None:
                if self._x is None and self._y is None:
                    self._x = [float(k) for k in x]
                    self._y = [float(k) for k in y]
                else:
                    self._x = [float(k) for k in self._x]
                    self._y = [float(k) for k in self._y]

                if self._upper_bound is None and self._lower_bound is None:
                    self._upper_bound = upper_bound
                    self._lower_bound = lower_bound

                fig, ax = plt.subplots()

                ax.grid(True)
                ax2 = ax2 = ax.secondary_xaxis('top', 
                                               functions=(self.radians_to_degrees, 
                                                          self.degrees_to_radians)
                                               )

                for i in range(1, 6):
                    poly_coeffs = np.polyfit(self._x, self._y, i)
                    poly_func = np.poly1d(poly_coeffs)
                    
                    if self._upper_bound is None and self._lower_bound is None:
                        x_plot = np.linspace(self._x[0], self._x[-1], 100)
                        y_plot = poly_func(x_plot)
                        ax.plot(x_plot, y_plot, label=f'Степень: {i}')
                    else:
                        x_plot = np.linspace(self._lower_bound, self._upper_bound, 100)
                        y_plot = poly_func(x_plot)
                        ax.plot(x_plot, y_plot, label=f'Степень: {i}')

                ax.scatter(self._x, self._y)
                ax.legend()
                ax.set_xlabel('x (Для тригонометрических функций - Радианы)')
                ax2.set_xlabel('x (Для тригонометрических функций - Градусы)')
                ax.set_ylabel('y')

                ax2.set_xlim(ax.get_xlim())

                param = len(os.listdir(path = '.'))
                self._figure_name = f'plot_{param}.png'
                plt.savefig(self._figure_name)
                plt.close('all')

            def get_plot_name(self) -> str:
                if self._figure_name is not None:
                    return self._figure_name


tmath = TeleMath()
models = TeleMath.Models()
telemath = TeleMath.Mathematica()

load_dotenv('KEYS.env')

OPENAI_TOKEN = os.getenv('AI_KEY')

def get_from_json(file_name: str, info: str):
    with open(file_name, 'r') as file:
        json_data = file.read()
    data = json.loads(json_data)

    result = data[info]

    file.close()

    return result

gpt_dictionary = get_from_json('texts-for-example.json', 'gpt_dict')

models.gpt(OPENAI_TOKEN, gpt_dictionary)
models.alpha_image(OPENAI_TOKEN)

BOT_TOKEN = os.getenv('BOT_KEY')
bot = Bot(BOT_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start'])
async def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("GPT")
    btn2 = types.KeyboardButton("DALL-E")
    btn3 = types.KeyboardButton("Mathematica")
    markup.add(btn1, btn2, btn3)

    info_about_modes = get_from_json('texts-for-example.json', 'bot_mode')

    await bot.send_message(message.chat.id, info_about_modes, reply_markup=markup)

@dp.message_handler(content_types=['text'])
async def main(message):
    if message.text == 'GPT':
        tmath.set_mode('GPT')
    elif message.text == 'DALL-E':
        tmath.set_mode('DALL-E')
    elif message.text == 'Mathematica':
        tmath.set_mode('Mathematica')

    if message.text == 'GPT' or tmath.mode == 'GPT':
        if tmath.mode == message.text:
            await bot.send_message(message.chat.id, 'Пришлите любой текстовый запрос.\n\nВ ответ Вам будет выслан текст сгенерированный GPT-3.5-turbo')

        if tmath.mode == 'GPT' and message.text != 'GPT':
            await bot.send_message(message.chat.id, models.generate_respose_gpt(message.text))
    elif message.text == 'DALL-E' or tmath.mode == 'DALL-E':
        if tmath.mode == message.text:
            await bot.send_message(message.chat.id, 'Пришлите любой текстовый запрос.\n\nВ ответ Вам будет выслано изображение сгенерированное DALL-E')

        if tmath.mode == 'DALL-E' and message.text != 'DALL-E':
            models.generate_respose_alpha(message.text)
            image = open('output.jpg', 'rb')

            await bot.send_photo(message.chat.id, image)

            image.close()
    elif message.text == 'Mathematica' or tmath.mode == 'Mathematica':
        if tmath.mode == message.text:
            math_info = get_from_json('texts-for-example.json', 'math_info')
            await bot.send_message(message.chat.id, math_info)

        if tmath.mode == 'Mathematica' and message.text != 'Mathematica':
            dict_result = {}
            parts = message.text.split(": ")
            dict_result['modificator'] = parts[0]
            dict_result['functions'] = parts[1]

            if dict_result['modificator'] == 'func':
                fig_name = telemath.plot_math_function(dict_result['functions'])
                derivates = telemath.derivate(dict_result['functions'])
                integrals = telemath.find_integral(dict_result['functions'])

                plot_image = open(fig_name, 'rb')
                await bot.send_photo(message.chat.id, plot_image)

                plot_image.close()
                os.remove(fig_name)

                functions = dict_result['functions'].split(';')
                for i in range(len(functions)):
                    msg = ''
                    msg += f'Введенная функция: {functions[i]}\n\n'
                    msg += f'Неопределенный интеграл: {integrals[i]}\n\n'
                    msg += f'Производная функции: {derivates[i]}\n\n'

                    await bot.send_message(message.chat.id, msg)
            elif dict_result['modificator'] == 'eval':
                data = dict_result['functions'].split(';')

                if len(data) == 2:
                    str_func = data[0]
                    str_values = data[1]

                    result = telemath.evaluate_expression(str_func, str_values)

                    optimised_result = []
                    for number in result:
                        optimised_result.append(telemath.optimize_number(number))
                    string = ''
                    for number in optimised_result:
                        string += f'{number}\n'

                    await bot.send_message(message.chat.id, string)
            elif dict_result['modificator'] == 'approx':
                data = dict_result['functions'].split(';')
                if len(data) == 2:
                    x = data[0].split()
                    y = data[1].split()
                    approx = TeleMath.Mathematica.Approximation()
                    approx.set_xy(x, y)
                    approx.approximate()
                    plot_image = open(approx.get_plot_name(), 'rb')

                    await bot.send_photo(message.chat.id, plot_image)

                    plot_image.close()
                    os.remove(approx.get_plot_name())
                elif len(data) == 3:
                    x = data[0].split()
                    y = data[1].split()
                    lower_b = float(data[2].split()[0])
                    upper_b = float(data[2].split()[1])

                    approx = TeleMath.Mathematica.Approximation()
                    approx.set_xy(x, y)
                    approx.set_bounds(lower_b, upper_b)
                    approx.approximate()
                    plot_image = open(approx.get_plot_name(), 'rb')

                    await bot.send_photo(message.chat.id, plot_image)

                    plot_image.close()
                    os.remove(approx.get_plot_name())
            elif dict_result['modificator'] == 'interp':
                data = dict_result['functions'].split(';')
                if len(data) == 2:
                    x = data[0].split()
                    y = data[1].split()
                    interp = TeleMath.Mathematica.Interpolation()
                    interp.set_xy(x, y)
                    interp.interpolate()
                    plot_image = open(interp.get_plot_name(), 'rb')

                    await bot.send_photo(message.chat.id, plot_image)

                    plot_image.close()
                    os.remove(interp.get_plot_name()) 
            elif dict_result['modificator'] == 'integ':
                data = dict_result['functions'].split(';')
                if len(data) == 2:
                    func = data[0]
                    borders = data[1].split() 
                    a = borders[0]
                    b = borders[1]

                    result = telemath.calculate_integral(func, a, b)
                    result = telemath.optimize_number(result)

                    await bot.send_message(message.chat.id, f'({func})dx = {result}')
            elif dict_result['modificator'] == 'dev':
                numbers = dict_result['functions'].split()
                numbers = [float(k) for k in numbers]
                result = telemath.calculate_std_dev(numbers)
                result = telemath.optimize_number(result)
                
                await bot.send_message(message.chat.id, f'std dev = {result}')

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
