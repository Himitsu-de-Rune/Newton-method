# Newton-method

Задача: Отыскание экстремума функции f(x) по всему протранству Rⁿ. Для определенности предположим, что требуется найти минимум функции f(x) в пространстве Rⁿ. 
  Допустим также, что f(x) обладает единственным минимумом и что все частные производные, по крайней мере первого порядка, существуют при любом значении x в Rⁿ.

Описание: Метод является прямым обобщением метода Ньютона отыскания корня уравнения
    g(t)=0, где g(t) - функция скалярной переменной. 
  Нас интересует n-мерная задача минимизации, где требуется определение корня уравнения ▽f(x)=0. 
  Применительно к определению корня уравнения ▽f(x)=0 разложение в ряд Тейлора дает 0 = ▽f(xᵏ⁺¹) ≈ ▽f(xᵏ) + H(xᵏ)(xᵏ⁺¹-xᵏ) 
    а последовательность {xᵏ} вырабатывается формулой xᵏ⁺¹ = xᵏ - H⁻¹(xᵏ)▽f(xᵏ), 
    при условии, что существует обратная матрица H⁻¹(xᵏ), где H - матрица Гессе.

Методика расчета:

  1) Определяем аналитические выражения (в символьном виде) для вычисления градиента рассматриваемой функции и квадратной матрицы Гессе:

![image](https://github.com/Himitsu-de-Rune/Newton-method/assets/170539653/b37b7c34-79de-4556-8f5f-165b7c79f0cb)

  2) Задаем начальное приближение X={x₁,...,xₙ}

    Далее выполняется итерационный процесс.

  3) Определяем новые значения аргументов функции

![image](https://github.com/Himitsu-de-Rune/Newton-method/assets/170539653/b8daba9c-7dfc-4166-9926-2f70b766c28a)

  4) Вычислительный процесс заканчивается, когда будет достигнута точка, в которой оценка градиента будет равна нулю. В противном случае возвращаемся к шагу 3 и продолжаем итерационный расчет.

Если функция f - квадратичная и равна f(x) = aᵀx + xᵀQx/2, где Q - положительно определенная матрица, 
  то, исходя из произвольной начальной точки x⁰, можно получить следущую точку: x¹ = x⁰ - Q⁻¹(a+Qx⁰) = -Q⁻¹a. 
Эта точка является точкой минимума функции f(x); таким образом, точка минимума может быть найдена за 1 шаг.

Однако этот метод требует затраты значительного времени на вычисление вторых производных функции f и обращение матрицы Гессе.
