This code allows you to create your own free WolframAlpha likeness directly in your Telegram bot. 
This code is given with an example of using the main class: TeleMath.
The TeleMath class has two subclasses: Models and Mathematica.

The first class, Models, implements the connection functionality of OpenAI models such as GPT-3.5-turbo and Image-Alpha-001.

The second class, Mathematica, implements the following functionality:
- derivate() - the derivative of a function
- find_integral() - indefinite integral
- plot_math_function() - plot the graph of the function
- evaluate_expression() - find the value of the function at specified points
- optimize_number() - optimize numbers to the form [ x*10^y, where x - number, rounded to two decimal places, y - power of ten ]
- plot_approx() - build approximating graph to the fifth power by Lagrange polynomial method
- plot_interp() - build the interpolating graph by the radial basis functions method (Rbf)
- calculate_integral() - find a definite integral
- calculate_std_dev() - find the standard deviation