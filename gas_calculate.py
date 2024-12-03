from dash import Dash, html, dash_table, dcc,html
from dash.dependencies import Input, Output,State
import dash_bootstrap_components as dbc
import math
import pandas as pd
import numpy as np
import plotly.graph_objs as go

# Инициализация приложения Dash
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Определение макета приложения
app.layout = html.Div([
    html.H4("Подбор диаметра НКТ по макросу Хилько"),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col(dcc.Input(id='input-pkr', type='number', placeholder='Введите значение', value=80,style={"width": "100%","margin-left": "20px"}),width=4),
                dbc.Col(html.Label('Pкр, ата', style={"width": "100%","margin-left": "20px"}),width=8)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Input(id='input-tkr', type='number', placeholder='Введите значение', value=211, style={"width": "100%","margin-left": "20px"}), width=4),
                dbc.Col(html.Label('Tкр, K', style={"width": "100%",'margin-left': '20px'}), width=8)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Input(id='input-roo', type='number', placeholder='Введите значение', value=0.64, style={"width": "100%","margin-left": "20px"}), width=4),
                dbc.Col(html.Label('Стн. плотность возд. (доля)', style={"width": "100%","margin-left": "20px"}), width=8)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Input(id='input-ppl', type='number', placeholder='Введите значение', value=277, style={"width": "100%","margin-left": "20px"}), width=4),
                dbc.Col(html.Label('Pпл, ата', style={"width": "100%","margin-left": "20px"}), width=8)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Input(id='input-tpl', type='number', placeholder='Введите значение', value=353, style={"width": "100%","margin-left": "20px"}), width=4),
                dbc.Col(html.Label('Tпл, K', style={"width": "100%","margin-left": "20px"}), width=8)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Input(id='input-tust', type='number', placeholder='Введите значение', value=295, style={"width": "100%","margin-left": "20px"}), width=4),
                dbc.Col(html.Label('Tуст, K', style={"width": "100%","margin-left": "20px"}), width=8)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Input(id='input-diam', type='number', placeholder='Введите значение', value=62, style={"width": "100%","margin-left": "20px"}), width=4),
                dbc.Col(html.Label('Внутр. Dнкт, мм', style={"width": "100%","margin-left": "20px"}), width=8)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Input(id='input-l', type='number', placeholder='Введите значение', value=2910, style={"width": "100%","margin-left": "20px"}), width=4),
                dbc.Col(html.Label('Глубина спуска НКТ, м', style={"width": "100%","margin-left": "20px"}), width=8)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Input(id='input-h', type='number', placeholder='Введите значение', value=2770, style={"width": "100%","margin-left": "20px"}), width=4),
                dbc.Col(html.Label('Глубина спуска НКТ (по вертикали), м', style={"width": "100%","margin-left": "20px"}), width=8)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Input(id='input-lq', type='number', placeholder='Введите значение', value=0.015, style={"width": "100%","margin-left": "20px"}), width=4),
                dbc.Col(html.Label('Эффективный λ НКТ', style={"width": "100%","margin-left": "20px"}), width=8)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Input(id='input-koefa', type='number', placeholder='Введите значение', value=168, style={"width": "100%","margin-left": "20px"}), width=4),
                dbc.Col(html.Label('Коэфф. фильтрации, ата²сут/тыс.м³ (a)', style={"width": "100%","margin-left": "20px"}), width=8)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Input(id='input-koefb', type='number', placeholder='Введите значение', value=0.035, style={"width": "100%","margin-left": "20px"}), width=4),
                dbc.Col(html.Label('Коэфф. фильтрации, ата²сут/тыс.м³ (b)', style={"width": "100%","margin-left": "20px"}), width=8)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Input(id='input-wgf', type='number', placeholder='Введите значение', value=165, style={"width": "100%","margin-left": "20px"}), width=4),
                dbc.Col(html.Label('ЖГФ, г/м3', style={"width": "100%","margin-left": "20px"}), width=8)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Input(id='input-rl', type='number', placeholder='Введите значение', value=750, style={"width": "100%","margin-left": "20px"}), width=4),
                dbc.Col(html.Label('Плотность жидкости, кг/м3', style={"width": "100%","margin-left": "20px"}), width=8)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Input(id='input-vmax', type='number', placeholder='Введите значение', value=20, style={"width": "100%","margin-left": "20px"}), width=4),
                dbc.Col(html.Label('Макс. допустимая скорость в НКТ, м/с', style={"width": "100%","margin-left": "20px"}), width=8)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Input(id='input-dpmax', type='number', placeholder='Введите значение', value=100, style={"width": "100%","margin-left": "20px"}), width=4),
                dbc.Col(html.Label('Макс. допустимая депрессия, ат', style={"width": "100%","margin-left": "20px"}), width=8)
            ], className="mb-3"),
        ], width=3),
        dbc.Col(dcc.Graph(id='output-graph'), width=9,)
    ]),
    dbc.Row(dbc.Col(dbc.Button('Обновить график', id='update-button', n_clicks=1, style={"margin-left": "20px"}), width=12), className="mb-3"),
    dbc.Row(dbc.Col(dbc.Button('Показать таблицу с расчетами', id='collapse-button-1', n_clicks=0, style={"margin-left": "20px"}), width=12), className="mb-3"),
    dbc.Col(
        dbc.Collapse([
                html.H4('Таблица с рассчитанными данными для графика'), 
                dcc.Loading(type="default",children=[html.Div(id='table-gas-lift-container')]) #вывод таблицы
                    ],is_open=False,id='collapse-1'),width=6,style={"width": "60%","margin-left": "20px","margin-right": "20px"})
])

def koefz(tpk, ppk, t, p):
    # tpk - критическая температура, К
    # ppk - критическое давление, ата
    # t - текущая температура, К
    # p - текущее давление, ата
    # Результат - возвращает значение коэффициента сжимаемости при p и t
    tp = t / tpk
    pp = p / ppk

    koefz_value = (0.4 * math.log(tp) / 2.303 + 0.73) ** pp + pp / 10
    return koefz_value

def p3_nns(ppkr, tpkr, ro, tp, py, qgas, v, rl, ty, d, l, h, lq):
    # ppkr - критическое давление, ата
    # tpkr - критическая температура, К
    # ro   - относ. плотность газа по воздуху
    # tp   - пластовая температура, К
    # py   - устьевое давление, ата
    # qgas - дебит газа, тыс.м3/сут
    # v    - дебит жидкости, м3/сут
    # rl   - плотность жидкости, кг/м3
    # ty   - температура на устье, К
    # d    - диаметр НКТ, мм
    # l    - длина трубы, м
    # h    - глубина скважины по вертикали, м
    # lq   - коэффициент гидравлического сопротивления НКТ
    # Результат - возвращает забойное давление в скважине, расчитанное по движ. столбу смеси воды и газа в НКТ для наклонной скважины, ата

    tc = tp
    qb = v / 1000
    mb = v * rl / 1000
    pit = py
    rn = ro * 1.205
    dcm = d / 10
    fu = 10000
    popr = h / l
    tsr = (tc - ty) / math.log(tc / ty)

    while fu > 0.01:
        psr = (py + pit) / 2
        kzsr = koefz(tpkr, ppkr, tsr, psr)
        rg = rn * psr * 293 / (1.033 * kzsr * tsr)
        qr = qgas * rn / rg

        fi = qr / (qr + qb)

        r = fi + (1 - fi) * rl / rg
        s = 0.03415 * ro * r * h / (kzsr * tsr)
        rsm = (qgas * rn + qb * rl) / (qgas + qb)
        gg = qgas * rn
        qsm = (gg + mb) / rsm

        p3a = math.sqrt(py ** 2 * math.exp(2 * s) + 1.377 * lq * kzsr ** 2 * tsr ** 2 * qsm ** 2 / (popr * r * dcm ** 5) * (math.exp(2 * s) - 1))

        fu = abs(pit - p3a)
        pit = p3a

    return p3a

def qab1(a, B, c, ppl, pzab):
    fun = 1000000
    qmin = -100000
    qmax = 100000
    iter = 0

    while abs(fun) > 0.1:
        iter += 1
        qi = (qmax + qmin) / 2
        fun = a * qi + B * qi ** 2 + c - ppl ** 2 + pzab ** 2
        if fun < 0:
            qmin = qi
        else:
            qmax = qi
        if iter > 2000:
            qi = 0
            break
    return qi

def rogas(ppk, tpk, ro, psr, tsr):
    rn = 1.205 * ro
    zsr = koefz(tpk, ppk, tsr, psr)
    rg = rn * psr * 293 / (1.033 * zsr * tsr)
    return rg

def py_nns(ppkr, tpkr, ro, tp, pz, qgas, v, rl, ty, d, l, h, lq):
    """
    ppkr - критическое давление, ата
    tpkr - критическая температура, К
    ro   - относ. плотность газа по воздуху
    tp   - пластовая температура, К
    pz   - забойное давление, ата
    qgas - дебит газа, тыс.м3/сут
    v    - дебит воды, м3/сут
    rl   - плотность жидкости, кг/м3
    ty   - температура на устье, К
    d    - диаметр НКТ, мм
    l    - длина трубы, м
    h    - глубина скважины по вертикали, м
    lq   - коэффициент гидравлического сопротивления НКТ
    Результат - возвращает забойное давление в скважине, рассчитанное по движ. столбу смеси воды и газа в НКТ для наклонной скважины, ата
    """

    try:
        tc = tp
        qb = v / 1000
        mb = v * rl / 1000
        pit = pz
        rn = ro * 1.205
        dcm = d / 10
        fu = 10000
        popr = h / l
        tsr = (tc - ty) / math.log(tc / ty)

        while fu > 0.01:
            psr = (pz + pit) / 2
            kzsr = koefz(tpkr, ppkr, tsr, psr)
            rg = rn * psr * 293 / (1.033 * kzsr * tsr)
            qr = qgas * rn / rg

            fi = qr / (qr + qb)

            r = fi + (1 - fi) * rl / rg
            s = 0.03415 * ro * r * h / (kzsr * tsr)

            rsm = (qgas * rn + qb * rl) / (qgas + qb)
            gg = qgas * rn
            qsm = (gg + mb) / rsm

            p3a = math.sqrt((pz ** 2 - 1.377 * lq * kzsr ** 2 * tsr ** 2 * qsm ** 2 / (popr * r * dcm ** 5) * (math.exp(2 * s) - 1)) / math.exp(2 * s))

            fu = abs(pit - p3a)
            pit = p3a

        return p3a
    except Exception as e:
        return -1

# Обработчик для обновления графика
@app.callback(
    [Output('output-graph', 'figure'),                  #Показ графика
     Output('table-gas-lift-container','children')],    #Показ таблицы
    [Input('update-button', 'n_clicks')],
    [State('input-pkr', 'value'),           # критическое давление, ата
     State('input-tkr', 'value'),           # критическая температура, К
     State('input-roo', 'value'),           # относ. плотность газа по воздуху
     State('input-ppl', 'value'),           # пластовая температура, К
     State('input-tpl', 'value'),           # пластовое давление, ата
     State('input-koefa', 'value'),         # пример значения коэффициента a
     State('input-koefb', 'value'),         # пример значения коэффициента b
     State('input-wgf', 'value'),           # пример значения wgf
     State('input-rl', 'value'),            # плотность жидкости, кг/м3
     State('input-tust', 'value'),          # температура на устье, К
     State('input-diam', 'value'),          # диаметр НКТ, мм
     State('input-l', 'value'),             # длина трубы, м
     State('input-h', 'value'),             # глубина скважины по вертикали, м
     State('input-lq', 'value'),            # коэффициент гидравлического сопротивления НКТ
     State('input-vmax', 'value'),          # Макс. допустимая скорость в НКТ, м/с
     State('input-dpmax', 'value')]         # макс Депрессия
)
def calculate_graph(n_click, pkr, tkr, roo, tpl, ppli, koefa, koefb, wgf, rl, ty, diam, l, h, lq, vmax, dpmax):
    if n_click<=0:
        return go.Figure(),None
    # Создаем DataFrame для хранения данных
    df = pd.DataFrame(columns=[
        'Q, тыс.м3/сут', 'Депр, атм', 'v, м/с', 'vmin, м/с', 'Py, атм',
        '(Pпл-Pу)/Q мод', 'Pз, атм', 'Доп. депрессия, атм', 'vmax, м/с',
        'dP/Q', 'Vбуф, м/с', 'Qводы, м3/сут'
    ])

    # Первая часть кода
    fff = 1000
    qmin = 0
    qmax = qab1(koefa, koefb, 0, ppli, 1.033)
    iter = 0

    while abs(fff) > 0.1:
        iter += 1
        qi = (qmin + qmax) / 2
        qw = qi * wgf / rl
        pzskv = p3_nns(pkr, tkr, roo, tpl, 1.033, qi, qw, rl, ty, diam, l, h, lq)
        pzpl = (ppli ** 2 - koefa * qi - koefb * qi ** 2) ** 0.5
        fff = pzskv - pzpl
        if fff > 0:
            qmax = qi
        else:
            qmin = qi
        if iter > 30:
            break

    kms = 15 / ((ppli - 1.033) / qi)

    qmax = qi * 0.99
    dq = qmax / 99
    qmin = 0

    # Вторая часть кода
    for j in range(1, 100):
        qi = qmin + j * dq
        pzpl = (ppli ** 2 - koefa * qi - koefb * qi ** 2) ** 0.5
        qw = qi * wgf / rl
        pyi = py_nns(pkr, tkr, roo, tpl, pzpl, qi, qw, rl, ty, diam, l, h, lq)

        if pyi < 1:
            break

        rg = rogas(pkr, tkr, roo, pzpl, tpl)
        rgy = rogas(pkr, tkr, roo, pyi, ty)

        vz = qi * 1.205 * roo / rg / 86.4 / (3.14 * (diam / 1000) ** 2 / 4)  #скорость на забое
        vy = qi * 1.205 * roo / rgy / 86.4 / (3.14 * (diam / 1000) ** 2 / 4) #скорость на утье
        vmin = 1.8 * 5.73 * (45 - 0.0455 * pzpl) ** 0.25 * pzpl ** -0.5      #мин допустимая скорость

        # Записываем данные в DataFrame
        df.loc[j] = [
            qi, 
            ppli - pzpl, 
            vz, 
            vmin, 
            pyi, 
            kms * (ppli - pyi) / qi,
            pzpl, #Рзаб
            dpmax, 
            vmax, 
            (ppli - pyi) / qi, 
            vy, #Скорость на устье
            qw
        ]

    # Создаем график с двумя осями ординат
    fig = go.Figure()

    # Основные данные
    fig.add_trace(
        go.Scatter(x=df['Q, тыс.м3/сут'], y=df['Депр, атм'], mode='lines', name='Депрессия, атм', line=dict(color='rgb(110,84,141)')))

    fig.add_trace(
        go.Scatter(x=df['Q, тыс.м3/сут'], y=df['Доп. депрессия, атм'], mode='lines', name='Доп. депрессия, атм', line=dict(color='rgb(61,150,174)', dash='dash')))

    fig.add_trace(
        go.Scatter(x=df['Q, тыс.м3/сут'], y=df['Py, атм'], mode='lines', name='Устьевое давление, атм', line=dict(color='rgb(247,150,70)'))) #оранж

    fig.add_trace(
        go.Scatter(x=df['Q, тыс.м3/сут'], y=df['Pз, атм'], mode='lines', name='Забойное давление, атм', line=dict(color='rgb(192,0,0)'))) #красный
    # Вспомогательные данные
    fig.add_trace(
        go.Scatter(x=df['Q, тыс.м3/сут'], y=df['vmin, м/с'], mode='lines', name='Мин.доп.скорость, м/с', line=dict(color='rgb(168,66,63)'), yaxis='y2'))

    fig.add_trace(
        go.Scatter(x=df['Q, тыс.м3/сут'], y=df['v, м/с'], mode='lines', name='Скорость газа на забое, м/с', line=dict(color='rgb(134,164,74)'), yaxis='y2'))

    fig.add_trace(
        go.Scatter(x=df['Q, тыс.м3/сут'], y=df['vmax, м/с'], mode='lines', name='Макс.доп.скорость, м/с', line=dict(color='rgb(65,111,166)', dash='dashdot'), yaxis='y2'))

    fig.add_trace(
        go.Scatter(x=df['Q, тыс.м3/сут'], y=df['Vбуф, м/с'], mode='lines', name='Скорость на устье, м/с', line=dict(color='rgb(206,142,141)'), yaxis='y2'))

    # Добавляем вторую ось ординат
    fig.update_layout(
        #title='График депрессии и скорости',
        height=800,
        #width=900,
        xaxis_title='</b>Дебит газа, тыс.м3/сут</b>',
        yaxis_title='</b>Давления, ата</b>',
        yaxis2=dict(
            title='</b>Скорость, м/с</b>',
            overlaying='y',
            side='right'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.3),
        margin=dict(l=0, r=160, t=0.1, b=0))  # Устанавливаем верхний отступ в 0)

    #собираем таблицу
    table = dash_table.DataTable(
    id='table-gas-lift-container-1',
    columns=[{"name": col, "id": col} for col in df.columns],
    data=df.round(2).to_dict('records'),
    style_table={'margin-bottom': '15px'},
    style_cell={'minWidth': '50px', 'width': '50px', #'maxWidth': '50px',
                #'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'textAlign':'left', #'padding': '10px',
                'backgroundColor': '#e6f7ff'},
    style_header={'backgroundColor': '#00a2e8','fontWeight': 'bold','color': 'white'})

    return fig,table

#фильтр таблицы МВР
@app.callback( 
    [Output("collapse-1", "is_open"),
     Output("collapse-button-1", "children")],
    [Input("collapse-button-1", "n_clicks")],
    [State("collapse-1", "is_open")])
def toggle_collapse(n_click, is_open):
    if n_click:
        if is_open:
            return False, "Показать таблицу с расчетами"
        else:
            return True, "Убрать таблицу с расчетами"
    return is_open, "Показать таблицу с расчетами"

if __name__ == '__main__': # Run the app
    app.run(debug=True) #True=dev mode False

# Запуск приложения
if __name__ == '__main__':
    app.run_server(debug=True)
