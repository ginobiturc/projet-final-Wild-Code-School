import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dash.exceptions import PreventUpdate
import base64

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler, QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN],suppress_callback_exceptions=True)

# Import df :
df_rein = pd.read_csv('df_ckd_7.csv')
df_foie = pd.read_csv('dataset_foie_ML.csv')
df_diabete = pd.read_csv('df_diabete.csv')
df_sein = pd.read_csv('dataset_sein_requilibre.csv')
df_coeur = pd.read_csv('df_coeur.csv')

# variables pour le machine learning
modelRein= ExtraTreesClassifier()
scal_rein= MaxAbsScaler()
modelFoie = ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                     criterion='gini', max_depth=35, max_features='sqrt',
                     max_leaf_nodes=None, max_samples=None,
                     min_impurity_decrease=0.0, min_samples_leaf=1,
                     min_samples_split=2, min_weight_fraction_leaf=0.0,
                     n_estimators=100, n_jobs=None, oob_score=False,
                     random_state=None, verbose=0, warm_start=False)
scal_foie = QuantileTransformer(n_quantiles= 609)
modelDiab = RandomForestClassifier(criterion = 'entropy', max_features = None, min_samples_leaf = 1, n_estimators = 15)
scal_diab = QuantileTransformer()
modelSein = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform')
scal_sein = QuantileTransformer()
modelCoeur = XGBClassifier(colsample_bytrr = 0.5, gamma = 0, learning_rate = 0.01, max_depth = 6, min_child_weight = 0)
scal_coeur = MaxAbsScaler()

# Info pour exporter en spreadsheet
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name('wildlab-b395b393ac77.json', scope)
gc = gspread.authorize(credentials)

# Layout du bandeau de navigation
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Accueil", href="/",active="exact",style={"font-size": "22px"})),
        dbc.NavItem(dbc.NavLink("Diabète", href="/diabete", active="exact",style={"font-size": "22px"})),
        dbc.NavItem(dbc.NavLink("Maladie Chronique Rénale", href="/maladie-chronique-renale", active="exact",style={"font-size": "22px"})),
        dbc.NavItem(dbc.NavLink("Foie", href="/foie", active="exact",style={"font-size": "22px"} )),
        dbc.NavItem(dbc.NavLink("Maladies cardiaques", href="/coeur", active="exact",style={"font-size": "22px"})),
        dbc.NavItem(dbc.NavLink("Cancer du sein", href="/sein", active="exact",style={"font-size": "22px"}))
    ],
    brand=dbc.NavbarBrand(html.Img(src="/assets/Capture_d_écran_2024-02-06_105619-removebg-preview.png", height="70px"), href="/"),
    color= 'primary',
    dark=True,
    className='fixed-top'
)

# Layout de la page d'accueil
accueil_layout = html.Div([
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H1("Bienvenue sur l’application d’aide au diagnostic InnoVie")),
    dbc.Row(html.Hr()),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row([html.P("Son principe ? Aider le soignant à déterminer si les indicateurs biochimiques du patient tendent "
                       "à indiquer la présence ou non d’une pathologie. "
                "Simple et rapide, il lui suffit  pour cela  de rentrer un certain nombre d’indicateurs biochimiques, "
                        "lesquels dépendent de la pathologie recherchée. "
                "L’équipe d’Innovie a travaillé d’arrache-pied sur ses modèles algorithmiques afin que les résultats "
                       "présentés soient aussi fiables que possible. "),
            html.P("Néanmoins, il est à souligner qu’il s’agit là d’un outil d’aide au diagnostic, qui ne doit en aucun cas"
                       " remplacer l’ensemble des investigations nécessaires à l’établissement d’un diagnostic par le praticien.")]
                , style={'text-align': 'center'}),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H1("Nos domaines d'expertise")),
    dbc.Row(html.Hr()),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardImg(src='/assets/diabete.png', top=True, style={"opacity": 0.7}),
                          dbc.CardImgOverlay(
                              dbc.CardBody(dbc.Button('DIABETE', color='pimary', href='/diabete',size='lg'),
                              style={"text-align": "center","display": "flex", "flex-direction": "column",
                                     "height": "100%","justify-content": "center"}))
                        ])),
        dbc.Col(dbc.Card([dbc.CardImg(src='/assets/rein.png', top=True, style={"opacity": 0.7}),
                          dbc.CardImgOverlay(
                              dbc.CardBody(dbc.Button('REINS', color='pimary', href='/maladie-chronique-renale',size='lg'),
                              style={"text-align": "center","display": "flex", "flex-direction": "column",
                                     "height": "100%","justify-content": "center"}))
                        ])),
        dbc.Col(dbc.Card([dbc.CardImg(src='/assets/foie.png', top=True, style={"opacity": 0.7}),
                          dbc.CardImgOverlay(
                              dbc.CardBody(dbc.Button('FOIE', color='pimary', href='/foie',size='lg'),
                              style={"text-align": "center","display": "flex", "flex-direction": "column",
                                     "height": "100%","justify-content": "center"}))
                        ])),
        dbc.Col(dbc.Card([dbc.CardImg(src='/assets/coeur.png', top=True, style={"opacity": 0.7}),
                          dbc.CardImgOverlay(
                              dbc.CardBody(dbc.Button('COEUR', color='pimary', href="/coeur",size='lg'),
                              style={"text-align": "center","display": "flex", "flex-direction": "column",
                                     "height": "100%","justify-content": "center"}))
                        ])),
        dbc.Col(dbc.Card([dbc.CardImg(src='/assets/sein.png', top=True, style={"opacity": 0.5}),
                          dbc.CardImgOverlay(
                              dbc.CardBody(dbc.Button('Cancer du sein', color='pimary', href='/sein',size='lg'),
                              style={"text-align": "center","display": "flex", "flex-direction": "column",
                                     "height": "100%","justify-content": "center"}))
                        ])),
        ], style={'text-align': 'center'}) ,
    dbc.Row(html.P(style={'margin-top': '20px'}))
],style={"margin": "0 20px", 'text-align': 'center'})

# Layout de la page diabète
diabete_layout = html.Div([
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H1("Diabète")),
    dbc.Row(html.Hr()),
    dbc.Row(html.Div([
        dbc.Button("Lexique des variables", id="open-offcanvas-diabete", n_clicks=0),
        dbc.Offcanvas(id="offcanvas-diabete", scrollable=True, title="Lexique des variables", is_open=False,
                    children=[ html.Div([
                html.H6("Glucose:", className="mt-3", style={"font-weight": "bold"}),
                html.P("Concentration de glucose plasmatique à 2 heures lors d'un test de tolérance au glucose oral."),

                html.H6("Pression artérielle:", className="mt-3", style={"font-weight": "bold"}),
                html.P("Pression diastolique (mm Hg). : la pression minimum enregistrée lors du relâchement du cœur."),
                html.P("Hypertension artérielle si >= 80 mmHg (mm de mercure)."),

                html.H6("Épaisseur du pli cutané du triceps:", className="mt-3", style={"font-weight": "bold"}),
                html.P("Epaisseur mesurée d’un pli de peau au niveau du triceps en mm."),

                html.H6("Insulin:", className="mt-3", style={"font-weight": "bold"}),
                html.P("Insuline sérique à 2 heures (mu U/ml)."),
                html.P("Les valeurs usuelles sont comprises entre 2 et 20 µUI/mL à jeun (14 à 140 pmol/L)."),
                html.P("Taux de glycémie normal 2h après le repas doit se situer entre 1,30 et 1,60."),
                html.P("La glycémie normale 2 h après le début d'un repas doit être inférieure à 1,40 g/L."),

                html.H6("IMC:", className="mt-3", style={"font-weight": "bold"}),
                html.P("Indice de masse corporelle (poids en kg/(taille en m)^2)."),
                html.P("IMC supérieur ou égal à 25 kg/m^2 = surpoids"),

                html.H6("Diabetes Pedigree Function:", className="mt-3", style={"font-weight": "bold"}),
                html.P("Une fonction qui évalue la probabilité de diabète en fonction des antécédents familiaux.")
            ])
        ],
    ),
])),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H3('Patient')),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(dbc.Col([dbc.InputGroup([
                dbc.InputGroupText('ID'),
                dbc.Input(id='Input_IDdiab', type='text',
                        placeholder= "Entrez l'ID du patient")])
                        ], className="d-grid gap-2 col-6 mx-auto")),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H3("Entrez vos informations ci dessous :")),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(dbc.Col([
            dbc.InputGroup([
                    dbc.InputGroupText('Age'),
                    dbc.Input(id='Input_agediab', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText('Nombre de grossesses'),
                    dbc.Input(id='Input_grossesse', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText('Indice de masse corporelle'),
                    dbc.Input(id='Input_imc', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText('Insuline sérique'),
                    dbc.Input(id='Input_insuline', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText("Glucose"),
                    dbc.Input(id='Input_glucose', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText('Epaisseur pli cutané du triceps'),
                    dbc.Input(id='Input_triceps', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText('Pression artérielle'),
                    dbc.Input(id='Input_pressart', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText("Diabete Pedigree Fonction"),
                    dbc.Input(id='Input_proba', type='number', placeholder= 'Entrez un nombre')
                ])
        ], className="d-grid gap-2 col-6 mx-auto")),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(dbc.Button(id='pred_diab_button', n_clicks=0, children='Résultat', outline=True,
                    className="d-grid gap-2 col-6 mx-auto btn-lg", color="dark")),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H3(id= 'diab_diag')),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H4(id= 'proba_diag_diab')),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H5(id = 'diab_export')),
    dbc.Row(html.P(style={'margin-top': '20px'}))
],style={"margin": "0 20px",'text-align': 'center'})

@app.callback(
    Output("offcanvas-diabete", "is_open"),
    Input("open-offcanvas-diabete", "n_clicks"),
    State("offcanvas-diabete", "is_open"),
)
def toggle_offcanvas_diabete(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(
    [Output('diab_diag', 'children'), Output('proba_diag_diab', 'children'), Output('diab_export', 'children')],
    [Input('pred_diab_button', 'n_clicks')],
    [State("Input_IDdiab", "value"),
     State('Input_agediab', 'value'),
     State('Input_grossesse', 'value'),
     State('Input_imc', 'value'),
     State('Input_insuline', 'value'),
     State('Input_glucose', 'value'),
     State('Input_triceps', 'value'),
     State('Input_pressart', 'value'),
     State('Input_proba', 'value'),
     ]
)

def diab_predict(n_clicks, input_IDdiab, input_agediab, input_grossesse, input_imc, input_insuline,
                 input_glucose, input_triceps, input_pressart, input_proba):
    if n_clicks == 0:
        raise PreventUpdate
    else:
        X = df_diabete[['Age','Pregnancies','BMI','Insulin', 'Glucose',
                     'SkinThickness','BloodPressure','DiabetesPedigreeFunction']]
        y = df_diabete['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size = 0.75)
        X_train_scal= scal_diab.fit_transform(X_train)
        X_test_scal= scal_diab.transform(X_test)
        modelDiab.fit(X_train_scal, y_train)
        diab_data = np.array([input_agediab, input_grossesse, input_imc, input_insuline,
                              input_glucose, input_triceps, input_pressart, input_proba]).reshape(1,8)
        diab_data = pd.DataFrame(diab_data, columns= ['Age','Pregnancies','BMI','Insulin', 'Glucose',
                     'SkinThickness','BloodPressure','DiabetesPedigreeFunction'])
        diab_data_scal = scal_diab.transform(diab_data)
        diab_predict = modelDiab.predict(diab_data_scal)
        current_date = datetime.datetime.now().strftime("%d-%m-%Y")
        diab_worksheet = gc.open('data_patient_diabete').sheet1
        diab_worksheet.append_row([current_date, input_IDdiab, input_agediab, input_grossesse, input_imc, input_insuline,
                 input_glucose, input_triceps, input_pressart, input_proba, int(diab_predict)])
        diab_export = f"Données exportées : id {input_IDdiab}"
        if diab_predict < 0.4:
            diab_diag = "Votre patient n'est pas malade"
            proba = modelDiab.predict_proba(diab_data_scal)
            diab_proba = proba[0,0]
            proba_diag_diab = f"Selon notre modèle, la fiabilité du résultat est évalué à  : {diab_proba*100} %"
            return diab_diag, proba_diag_diab, diab_export
        else :
            diab_diag = 'Il est possible que votre patient soit malade, veuillez approfondir les analyses'
            proba = modelDiab.predict_proba(diab_data_scal)
            diab_proba = proba[0,1]
            proba_diag_diab = f"Selon notre modèle, la fiabilité du résultat est évalué à  : {diab_proba*100} %"
            return diab_diag, proba_diag_diab, diab_export



# Layout de la page maladie chronique rénale
mcr_layout = html.Div([
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H1("Maladie Chronique Rénale")),
    dbc.Row(html.Hr()),
    dbc.Row(html.Div([
        dbc.Button("Lexique des variables", id="open-offcanvas-rein", n_clicks=0),
        dbc.Offcanvas(id="offcanvas-rein", scrollable=True, title="Lexique des variables", is_open=False,
                    children=[ html.Div([
                html.H6("Volume globule rouge:", className="mt-3", style={"font-weight": "bold"}),
                html.P("Lorsque vos globules rouges sont en nombre et en consistance normale,"
                       "votre taux de VGM se situe entre 80 et 100 femtolitres"),

                html.H6("Pression artérielle (mmHg):", className="mt-3", style={"font-weight": "bold"}),
                html.P("une tension de 130/80 ou plus est considérée comme élevée"),
            ])
        ],
    ),
])),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H3('Patient')),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row([
        dbc.Col(html.P(' ')),
        dbc.Col(dbc.InputGroup([
                dbc.InputGroupText('ID'),
                dbc.Input(id='Input_IDrein', type='text',
                        placeholder= "Entrez l'ID du patient")])),
        dbc.Col(html.P(' '))
        ]),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H3("Entrez vos informations ci dessous :")),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(dbc.Col([
            dbc.InputGroup([
                    dbc.InputGroupText('Age'),
                    dbc.Input(id='Input_age', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText('Volume de globules rouges'),
                    dbc.Input(id='Input_pcv', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText("Pression artérielle"),
                    dbc.Input(id='Input_pressure', type='number', placeholder= 'Entrez un nombre')
                ]),
    ], className="d-grid gap-2 col-6 mx-auto")),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(dbc.Col([
            dbc.Label("Niveau de sucre"),
            dcc.Slider(
                id='slider_sucre',
                min=0,
                max=5,
                step=1,
                marks={i: str(i) for i in range(6)},
                value=0)
    ], className="d-grid gap-2 col-6 mx-auto")),
    dbc.Row(html.P(' ')),
    dbc.Row([dbc.Card([dbc.CardHeader("Maladie coronarienne ? ",style={"font-size": "18px"}),
            dbc.CardBody([dbc.RadioItems(id='Input_coronary',
                               options=[{"label": "Oui", "value": 1},
                                        {"label": "Non", "value": 0}],
                                inputStyle={'border-color': 'grey'})])
                 ],className="d-grid gap-2 col-4 mx-auto")
    ]),
    dbc.Row(html.P(' ')),
    dbc.Row([dbc.Card([dbc.CardHeader("Hypertension ? ",style={"font-size": "18px"}),
            dbc.CardBody([dbc.RadioItems(id='Input_hpt',
                               options=[{"label": "Oui", "value": 1},
                                        {"label": "Non", "value": 0}],
                                inputStyle={'border-color': 'grey'})])
                 ],className="d-grid gap-2 col-4 mx-auto")
    ]),
    dbc.Row(html.P(' ')),
    dbc.Row([dbc.Card([dbc.CardHeader("Diabète ?  ",style={"font-size": "18px"}),
            dbc.CardBody([dbc.RadioItems(id='Input_dbt',
                               options=[{"label": "Oui", "value": 1},
                                        {"label": "Non", "value": 0}],
                                inputStyle={'border-color': 'grey'})])
                 ],className="d-grid gap-2 col-4 mx-auto")
    ]),
    dbc.Row(html.P(' ')),
    dbc.Row([dbc.Card([dbc.CardHeader("Cellules de pus ? ",style={"font-size": "18px"}),
            dbc.CardBody([dbc.RadioItems(id='Input_pus',
                               options=[{"label": "Présentes", "value": 1},
                                        {"label": "Non présentes", "value": 0}],
                                inputStyle={'border-color': 'grey'})])
                 ],className="d-grid gap-2 col-4 mx-auto")
    ]),
    dbc.Row(html.P(' ')),
    dbc.Row([dbc.Card([dbc.CardHeader("Anémie ?  ",style={"font-size": "18px"}),
            dbc.CardBody([dbc.RadioItems(id='Input_anemie',
                               options=[{"label": "Oui", "value": 1},
                                        {"label": "Non", "value": 0}],
                                inputStyle={'border-color': 'grey'})])
                 ],className="d-grid gap-2 col-4 mx-auto")
    ]),
    dbc.Row(html.P(' ')),
    dbc.Row([dbc.Card([dbc.CardHeader("Appétit ?  ",style={"font-size": "18px"}),
            dbc.CardBody([dbc.RadioItems(id='Input_appet',
                               options=[{"label": "Bon", "value": 0},
                                        {"label": "Mauvais", "value": 1}],
                                inputStyle={'border-color': 'grey'})])
                 ],className="d-grid gap-2 col-4 mx-auto")
    ]),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(dbc.Button(id='pred_rein_button', n_clicks=0, children='Résultat', outline=True,
                    className="d-grid gap-2 col-6 mx-auto btn-lg", color="dark")),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H3(id= 'rein_diag')),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H4(id= 'proba_diag_rein')),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H5(id = 'rein_export')),
    dbc.Row(html.P(style={'margin-top': '20px'}))
],style={"margin": "0 20px",'text-align': 'center'})

@app.callback(
    Output("offcanvas-rein", "is_open"),
    Input("open-offcanvas-rein", "n_clicks"),
    State("offcanvas-rein", "is_open"),
)
def toggle_offcanvas_rein(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(
    [Output('rein_diag', 'children'),Output('proba_diag_rein', 'children'), Output('rein_export', 'children')],
    [Input('pred_rein_button', 'n_clicks')],
    [State('Input_IDrein', "value"),
     State('Input_age', 'value'),
     State('Input_pcv', 'value'),
     State('Input_pressure', 'value'),
     State('slider_sucre', 'value'),
     State('Input_coronary', 'value'),
     State('Input_hpt', 'value'),
     State('Input_dbt', 'value'),
     State('Input_pus', 'value'),
     State('Input_anemie', 'value'),
     State('Input_appet', 'value'),
     ]
)

def rein_predict(n_clicks, input_IDrein, input_age, input_pcv, input_pressure, slider_sucre,
                 input_coronary, input_hpt, input_dbt, input_pus, input_anemie, input_appet):
    if n_clicks == 0:
        raise PreventUpdate
    else:
        X = df_rein[['blood_pressure', 'sugar', 'age', 'hypertension', 'diabete', 'pus_cell', 'appet', 'coronary_artery_disease',
        'anemie', 'packed_cell_vol']]
        y = df_rein['classification']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size = 0.75)
        X_train_scal= scal_rein.fit_transform(X_train)
        X_test_scal= scal_rein.transform(X_test)
        modelRein.fit(X_train_scal, y_train)
        rein_data = np.array([input_age, input_pcv, input_pressure, slider_sucre,
                 input_coronary, input_hpt, input_dbt, input_pus, input_anemie, input_appet]).reshape(1,10)
        rein_data = pd.DataFrame(rein_data, columns= ['blood_pressure', 'sugar', 'age', 'hypertension', 'diabete',
                                                      'pus_cell', 'appet', 'coronary_artery_disease','anemie', 'packed_cell_vol'])
        rein_data_scal = scal_rein.transform(rein_data)
        rein_predict = modelRein.predict(rein_data_scal)
        current_date = datetime.datetime.now().strftime("%d-%m-%Y")
        rein_worksheet = gc.open('data_patient_rein').sheet1
        rein_worksheet.append_row([current_date, input_IDrein, input_age, input_pcv, input_pressure, slider_sucre,
                 input_coronary, input_hpt, input_dbt, input_pus, input_anemie, input_appet, int(rein_predict)])
        rein_export = f"Données exportées : id {input_IDrein}"
        if rein_predict < 0.4:
            rein_diag = "Votre patient n'est pas malade"
            proba = modelRein.predict_proba(rein_data_scal)
            rein_proba = proba[0,0]
            proba_diag_rein = f"Selon notre modèle, la fiabilité du résultat est évalué à  : {rein_proba*100} %"
            return rein_diag, proba_diag_rein, rein_export
        else :
            rein_diag = 'Il est possible que votre patient soit malade, veuillez approfondir les analyses'
            proba = modelRein.predict_proba(rein_data_scal)
            rein_proba = proba[0,1]
            proba_diag_rein = f"Selon notre modèle, la fiabilité du résultat est évalué à  : {rein_proba*100} %"
            return rein_diag,proba_diag_rein, rein_export


# Layout de la page du foie
foie_layout = html.Div([
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H1("Maladie Chronique du foie")),
    dbc.Row(html.Hr()),
    dbc.Row(html.Div([
        dbc.Button("Lexique des variables", id="open-offcanvas-foie", n_clicks=0),
        dbc.Offcanvas(id="offcanvas-foie", scrollable=True, title="Lexique des variables", is_open=False,
                    children=[ html.Div([
                html.H6("Alamine Aminotransferase:", className="mt-3", style={"font-weight": "bold"}),
                html.P('les transaminases ALAT, également appelées TGP ou SGPT'),
                html.P('Le taux d’ALAT normal entre 8 et 35 (UI/L) pour les hommes et entre 6 et 25 UI/L pour les femmes.'),

                html.H6("Alkaline Phosphotase:", className="mt-3", style={"font-weight": "bold"}),
                html.P("Les phosphatases alcalines sont élevées chez les enfants en raison de la croissance osseuse due à l’âge et chez la femme enceinte"),
                html.P("Sinon le taux normal d'une femme : 35 à 104 U/L et d'un homme : 40 à 129 U/L"),

                html.H6("Albumin and Globulin Ratio:", className="mt-3", style={"font-weight": "bold"}),
                html.P("Rapport de l'albumine à la globuline dans le sang."),
                html.P('Le rapport albumine/globuline se situe entre 1,2 et 1,8.'),

                html.H6("Bilirubine totale dans le sang:", className="mt-3", style={"font-weight": "bold"}),
                html.P("A l'âge adulte le taux normal est compris entre 3 et 10 mg/L"),
            ])
        ],
    ),
])),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H3('Patient')),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row([
        dbc.Col(html.P(' ')),
        dbc.Col(dbc.InputGroup([
                dbc.InputGroupText('ID'),
                dbc.Input(id='Input_IDfoie', type='text',
                        placeholder= "Entrez l'ID du patient")])),
        dbc.Col(html.P(' '))
        ]),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H3("Entrez vos informations ci dessous :")),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row([
        dbc.Col([dbc.InputGroup([
                    dbc.InputGroupText('Age '),
                    dbc.Input(id='Input_agefoie', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText('Alamine Aminotransferase'),
                    dbc.Input(id='Input_alamine', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText('Alkaline Phosphotase'),
                    dbc.Input(id='Input_alka', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText('Ratio Albumine/Globuline'),
                    dbc.Input(id='Input_albglo', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText('Bilirubin'),
                    dbc.Input(id='Input_bili', type='number', placeholder= 'Entrez un nombre')
                ])], className="d-grid gap-2 col-6 mx-auto")
    ]),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(dbc.Button(id='pred_foie_button', n_clicks=0, children='Résultat', outline=True,
                    className="d-grid gap-2 col-6 mx-auto btn-lg", color="dark")),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H3(id= 'foie_diag')),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H4(id= 'proba_diag_foie')),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H5(id = 'foie_export')),
    dbc.Row(html.P(style={'margin-top': '20px'}))
],style={"margin": "0 20px",'text-align': 'center'})

@app.callback(
    Output("offcanvas-foie", "is_open"),
    Input("open-offcanvas-foie", "n_clicks"),
    State("offcanvas-foie", "is_open"),
)
def toggle_offcanvas_foie(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(
    [Output('foie_diag', 'children'),Output('proba_diag_foie', 'children'), Output('foie_export', 'children')],
    [Input('pred_foie_button', 'n_clicks')],
    [State('Input_IDfoie', 'value'),
     State('Input_agefoie', 'value'),
     State('Input_alamine', 'value'),
     State('Input_alka', 'value'),
     State('Input_albglo', 'value'),
     State('Input_bili', 'value'),
     ]
)

def foie_predict(n_clicks,input_IDfoie, input_agefoie, input_alamine, input_alka, input_albglo, input_bili):
    if n_clicks == 0:
        raise PreventUpdate
    else:
        X = df_foie[['Age','Alamine_Aminotransferase','Alkaline_Phosphotase',
                     'Albumin_and_Globulin_Ratio','Total_Bilirubin']]
        y = df_foie['Dataset']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size = 0.75)
        X_train_scal= scal_foie.fit_transform(X_train)
        X_test_scal= scal_foie.transform(X_test)
        modelFoie.fit(X_train_scal, y_train)
        foie_data = np.array([input_agefoie, input_alamine, input_alka, input_albglo, input_bili]).reshape(1,5)
        foie_data = pd.DataFrame(foie_data, columns= ['Age','Alamine_Aminotransferase','Alkaline_Phosphotase',
                     'Albumin_and_Globulin_Ratio','Total_Bilirubin'])
        foie_data_scal = scal_foie.transform(foie_data)
        foie_predict = modelFoie.predict(foie_data_scal)
        current_date = datetime.datetime.now().strftime("%d-%m-%Y")
        foie_worksheet = gc.open('data_patient_foie').sheet1
        foie_worksheet.append_row([current_date,input_IDfoie, input_agefoie, input_alamine, input_alka, input_albglo, input_bili, int(foie_predict)])
        foie_export = f"Données exportées : id {input_IDfoie}"
        if foie_predict < 0.4:
            foie_diag = "Votre patient n'est pas malade"
            proba = modelFoie.predict_proba(foie_data_scal)
            foie_proba = proba[0,0]
            proba_diag_foie = f"Selon notre modèle, la fiabilité du résultat est évalué à  : {foie_proba*100} %"
            return foie_diag, proba_diag_foie, foie_export
        else :
            foie_diag = 'Il est possible que votre patient soit malade, veuillez approfondir les analyses'
            proba = modelFoie.predict_proba(foie_data_scal)
            foie_proba = proba[0,1]
            proba_diag_foie = f"Selon notre modèle, la fiabilité du résultat est évalué à  : {foie_proba*100} %"
            return foie_diag, proba_diag_foie, foie_export


# Layout de la page du coeur
coeur_layout = html.Div([
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H1("Maladies cardiaques")),
    dbc.Row(html.Hr()),
    dbc.Row(html.Div([
        dbc.Button("Lexique des variables", id="open-offcanvas-coeur", n_clicks=0),
        dbc.Offcanvas(id="offcanvas-coeur", scrollable=True, title="Lexique des variables", is_open=False,
                    children=[ html.Div([
                html.H6("Pression artérielle:", className="mt-3", style={"font-weight": "bold"}),
                html.P("Pression diastolique (mm Hg). : la pression minimum enregistrée lors du relâchement du cœur."),
                html.P("Hypertension artérielle si >= 80 mmHg (mm de mercure)."),

                html.H6("Cholesterol:", className="mt-3", style={"font-weight": "bold"}),
                html.P("Cholesterol (chol, mg/dl)"),
                html.P("Le taux de cholestérol total = les valeurs du HDL + les valeurs du LDL."),

                html.H6("Glycémie", className="mt-3", style={"font-weight": "bold"}),
                html.P("Glycémie (fbs,  > 120 mg/dl, 1 = oui, 0 = non)"),

                html.H6("Fréquence cardiaque max : ", className="mt-3", style={"font-weight": "bold"}),
                html.P("Fréquence cardiaque maximale atteinte (thalach)"),

                html.H6("Type de thalassémie :", className="mt-3", style={"font-weight": "bold"}),
                html.P("Type de thalassémie (thal, 0 = normal, 2 = défaut fixe, 3 = défaut réversible)"),

                html.H6("Depression segment ST", className="mt-3", style={"font-weight": "bold"}),
                html.P("Dépression du segment ST induite par l'exercice par rapport au repos (oldpeak)"),
                html.P("ST se rapporte aux positions sur le tracé de l'ECG"),

                html.H6("Pente segment ST :", className="mt-3", style={"font-weight": "bold"}),
                html.P("Pente du segment ST à l'effort maximal (slope)"),
                html.P("1 = pente ascendante, 2 = plat, 3 = pente descendante"),

                html.H6("Nombre de vaisseaux majeurs colorés par fluoroscopie", className="mt-3", style={"font-weight": "bold"}),
                html.P("Nombre de vaisseaux majeurs colorés par fluoroscopie (ca)"),

                html.H6("exang :", className="mt-3", style={"font-weight": "bold"}),
                html.P("Angine de poitrine provoquée par l'exercice (exang), oui = 1, non = 0"),

                html.H6("Résultats électrocardiogrammes :", className="mt-3", style={"font-weight": "bold"}),
                html.P("Résultats électrocardiographiques au repos (restecg)"),
                html.P("normal= 0, présentant une anomalie de l'onde ST-T = 1"),
                html.P("présentant une hypertrophie ventriculaire gauche probable ou certaine selon les critères d'Estes = 2"),

                html.H6("Douleur thoracique:", className="mt-3", style={"font-weight": "bold"}),
                html.P("Douleur thoracique(cp)"),
                html.P(": angine de poitrine typique, 2 = angine de poitrine atypique"),
                html.P("3 =  douleur non liée à une angine de poitrine, 4 = asymptomatique")
            ])
        ],
    ),
])),

    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H3('Patient')),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row([
        dbc.Col(html.P(' ')),
        dbc.Col(dbc.InputGroup([
                dbc.InputGroupText('ID'),
                dbc.Input(id='Input_IDcoeur', type='text',
                        placeholder= "Entrez l'ID du patient")])),
        dbc.Col(html.P(' '))
        ]),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H3("Entrez vos informations ci dessous :")),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row([dbc.Card([dbc.CardHeader("Sexe ",style={"font-size": "18px"}),
            dbc.CardBody([dbc.RadioItems(id='Input_genre',
                           options=[{"label": "\u2640"+ " "+"Femme", "value": 1},
                                    {"label": "\u2642" + " "+"Homme", "value": 0}],
                            inputStyle={'border-color': 'grey'})])
                 ],className="d-grid gap-2 col-4 mx-auto")
    ]),
    dbc.Row(html.P(" ")),
    dbc.Row([
        dbc.Col([dbc.InputGroup([
                    dbc.InputGroupText("Age"),
                    dbc.Input(id='Input_agecoeur', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText('Pression artérielle'),
                    dbc.Input(id='Input_pad', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText('Cholesterol'),
                    dbc.Input(id='Input_chol', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText("Glycémie"),
                    dbc.Input(id='Input_glycemie', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText("Fréquence cardiaque max"),
                    dbc.Input(id='Input_freqmax', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText("Type de Thalassémie"),
                    dbc.Input(id='Input_thalass', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText("Dépression Segment ST"),
                    dbc.Input(id='Input_depression', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText("Pente Segment ST"),
                    dbc.Input(id='Input_pente', type='number', placeholder= 'Entrez un nombre')
                ]),
            dbc.InputGroup([
                    dbc.InputGroupText("Nb vaisseaux majeurs colorés par fluoroscopie"),
                    dbc.Input(id='Input_vaisseaux', type='number', placeholder= 'Entrez un nombre')
                ])],className="d-grid gap-2 col-6 mx-auto")
    ]),
    dbc.Row(html.P(" ")),
    dbc.Row([dbc.Card([dbc.CardHeader("Exang ",style={"font-size": "18px"}),
            dbc.CardBody([dbc.RadioItems(id='Input_exang',
                               options=[{"label": "Non", "value": 0},
                                        {"label": "Oui", "value": 1}],
                                inputStyle={'border-color': 'grey'})])
                 ],className="d-grid gap-2 col-4 mx-auto")
    ]),
    dbc.Row(html.P(" ")),
    dbc.Row([dbc.Card([dbc.CardHeader("Résultats électrocardiogrammes ",style={"font-size": "18px"}),
            dbc.CardBody([dbc.RadioItems(id='Input_electro',
                               options=[{"label": "Normal", "value": 0},
                                        {"label": "Anomalie", "value": 1},
                                        {"label": "hypertrophie ventriculaire", "value": 2}],
                                inputStyle={'border-color': 'grey'})])
                 ],className="d-grid gap-2 col-4 mx-auto")
    ]),
    dbc.Row(html.P(" ")),
    dbc.Row([dbc.Card([dbc.CardHeader("Douleur thoracique ",style={"font-size": "18px"}),
            dbc.CardBody([dbc.RadioItems(id='Input_thoracique',
                               options=[{"label": "Angine de poitrine typique", "value": 1},
                                        {"label": "Angine de poitrine atypique", "value": 2},
                                        {"label": "douleur non liée à une angine de poitrine", "value": 3},
                                        {"label": "Asymptomatique", "value": 4}],
                                inputStyle={'border-color': 'grey'})])
                 ],className="d-grid gap-2 col-4 mx-auto")
    ]),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(dbc.Button(id='pred_coeur_button', n_clicks=0, children='Résultat', outline=True,
                    className="d-grid gap-2 col-6 mx-auto btn-lg", color="dark")),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H3(id= 'coeur_diag')),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H4(id= 'proba_diag_coeur')),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H5(id = 'coeur_export')),
    dbc.Row(html.P(style={'margin-top': '20px'})),
],style={"margin": "0 20px", 'text-align': 'center'})

@app.callback(
    Output("offcanvas-coeur", "is_open"),
    Input("open-offcanvas-coeur", "n_clicks"),
    State("offcanvas-coeur", "is_open"),
)
def toggle_offcanvas_coeur(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(
    [Output('coeur_diag', 'children'), Output('proba_diag_coeur', 'children'), Output('coeur_export', 'children')],
    [Input('pred_coeur_button', 'n_clicks')],
    [State('Input_IDcoeur', "value"),
     State('Input_genre', 'value'),
     State('Input_agecoeur', 'value'),
     State('Input_pad', 'value'),
     State('Input_chol', 'value'),
     State('Input_glycemie', 'value'),
     State('Input_freqmax', 'value'),
     State('Input_thalass', 'value'),
     State('Input_depression', 'value'),
     State('Input_pente', 'value'),
     State('Input_vaisseaux', 'value'),
     State('Input_exang', 'value'),
     State('Input_electro', 'value'),
     State('Input_thoracique', 'value')
     ]
)

def coeur_predict(n_clicks, input_IDcoeur, input_genre, input_agecoeur, input_pad, input_chol, input_glycemie, input_freqmax, input_thalass,
                 input_depression, input_pente, input_vaisseaux, input_exang, input_electro, input_thoracique):
    if n_clicks == 0:
        raise PreventUpdate
    else:
        X = df_coeur[['sex', 'age', 'trestbps', 'chol', 'fbs', 'thalach', 'thal', 'oldpeak', 'slope', 'ca', 'exang', 'restecg','cp']]
        y = df_coeur['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size = 0.75)
        X_train_scal= scal_coeur.fit_transform(X_train)
        X_test_scal= scal_coeur.transform(X_test)
        modelCoeur.fit(X_train_scal, y_train)
        coeur_data = np.array([input_genre, input_agecoeur, input_pad, input_chol, input_glycemie, input_freqmax, input_thalass,
                 input_depression, input_pente, input_vaisseaux, input_exang, input_electro, input_thoracique]).reshape(1,13)
        coeur_data = pd.DataFrame(coeur_data, columns= ['sex', 'age', 'trestbps', 'chol', 'fbs', 'thalach', 'thal',
                                                         'oldpeak', 'slope', 'ca', 'exang', 'restecg','cp'])
        coeur_data_scal = scal_coeur.transform(coeur_data)
        coeur_predict = modelCoeur.predict(coeur_data_scal)
        current_date = datetime.datetime.now().strftime("%d-%m-%Y")
        coeur_worksheet = gc.open('data_patient_coeur').sheet1
        coeur_worksheet.append_row([current_date, input_IDcoeur, input_genre, input_agecoeur, input_pad, input_chol, input_glycemie, input_freqmax, input_thalass,
                 input_depression, input_pente, input_vaisseaux, input_exang, input_electro, input_thoracique, int(coeur_predict)])
        coeur_export = f"Données exportées : id {input_IDcoeur}"
        if coeur_predict < 0.4:
            coeur_diag = "Votre patient n'est pas malade"
            proba = modelCoeur.predict_proba(coeur_data_scal)
            coeur_proba = proba[0,0]
            proba_diag_coeur = f"Selon notre modèle, la fiabilité du résultat est évalué à  : {coeur_proba*100} %"
            return coeur_diag, proba_diag_coeur, coeur_export
        else :
            coeur_diag = 'Il est possible que votre patient soit malade, veuillez approfondir les analyses'
            proba = modelCoeur.predict_proba(coeur_data_scal)
            coeur_proba = proba[0,1]
            proba_diag_coeur = f"Selon notre modèle, la fiabilité du résultat est évalué à  : {coeur_proba*100} %"
            return coeur_diag, proba_diag_coeur, coeur_export


# Layout de la page du cancer du sein
sein_layout = html.Div([
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H1("Cancer du Sein", style={'text-align': 'center'})),
    dbc.Row(html.Hr()),
    dbc.Row(html.Div([
        dbc.Button("Lexique des variables", id="open-offcanvas-sein", n_clicks=0),
        dbc.Offcanvas(id="offcanvas-sein", scrollable=True, title="Lexique des variables", is_open=False,
                    children=[ html.Div([
                html.H6("Texture Mean:", className="mt-3", style={"font-weight": "bold"}),
                html.P("Moyenne de l'homogénéité interne des cellules"),

                html.H6("Area Mean:", className="mt-3", style={"font-weight": "bold"}),
                html.P('Moyenne de la surface des cellules'),

                html.H6("Smoothness Mean:", className="mt-3", style={"font-weight": "bold"}),
                html.P('Moyenne de la régularité des contours des cellules'),

                html.H6("Compactness mean:", className="mt-3", style={"font-weight": "bold"}),
                html.P("Moyenne de la compacité de la tumeur"),

                html.H6("Concavity Mean:", className="mt-3", style={"font-weight": "bold"}),
                html.P("Moyenne de la gravité des parties concaves du contour"),

                html.H6("Concave points mean:", className="mt-3", style={"font-weight": "bold"}),
                html.P("Moyenne du nombre de parties concaves du contour de"),

                html.H6("Symmetry Mean:", className="mt-3", style={"font-weight": "bold"}),
                html.P("Moyenne des symétries de la tumeur"),
            ])
        ],
    ),
])),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H3('Patient')),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row([
        dbc.Col(html.P(' ')),
        dbc.Col(dbc.InputGroup([
                dbc.InputGroupText('ID'),
                dbc.Input(id='Input_IDsein', type='text',
                        placeholder= "Entrez l'ID du patient")])),
        dbc.Col(html.P(' '))
        ]),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H3("Entrez vos informations sur la cellule ci-dessous :")),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.InputGroupText('Texture Mean :'),
                dbc.Input(id='Input_texture_mean', type='number', placeholder='Entrez un nombre')
            ]),
            dbc.InputGroup([
                dbc.InputGroupText('Area mean :'),
                dbc.Input(id='Input_area_mean', type='number', placeholder='Entrez un nombre')
            ]),
            dbc.InputGroup([
                dbc.InputGroupText('Smoothness mean :'),
                dbc.Input(id='Input_smoothness_mean', type='number', placeholder='Entrez un nombre')
            ]),
            dbc.InputGroup([
                dbc.InputGroupText('Compactness mean :'),
                dbc.Input(id='Input_compactness_mean', type='number', placeholder='Entrez un nombre')
            ]),
            dbc.InputGroup([
                dbc.InputGroupText('Concavity mean :'),
                dbc.Input(id='Input_concavity_mean', type='number', placeholder='Entrez un nombre')
            ]),
            dbc.InputGroup([
                dbc.InputGroupText('Concave points mean :'),
                dbc.Input(id='Input_concave_points_mean', type='number', placeholder='Entrez un nombre')
            ]),
            dbc.InputGroup([
                dbc.InputGroupText('Symmetry mean :'),
                dbc.Input(id='Input_symmetry_mean', type='number', placeholder='Entrez un nombre')
            ]),
        ], className="d-grid gap-2 col-6 mx-auto"),
    ]),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(dbc.Button(id='pred_sein_button', n_clicks=0, children='Résultat', outline=True,
                       className="d-grid gap-2 col-6 mx-auto btn-lg", color="dark")),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H3(id='sein_diag')),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H4(id='proba_diag_sein')),
    dbc.Row(html.P(style={'margin-top': '20px'})),
    dbc.Row(html.H5(id = 'sein_export')),
    dbc.Row(html.P(style={'margin-top': '20px'}))
],style={"margin": "0 20px",'text-align': 'center'})

@app.callback(
    Output("offcanvas-sein", "is_open"),
    Input("open-offcanvas-sein", "n_clicks"),
    State("offcanvas-sein", "is_open"),
)
def toggle_offcanvas_sein(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(
    [Output('sein_diag', 'children'), Output('proba_diag_sein', 'children'), Output('sein_export', 'children')],
    [Input('pred_sein_button', 'n_clicks')],
    [State('Input_IDsein','value'),
     State('Input_texture_mean', 'value'),
     State('Input_area_mean', 'value'),
     State('Input_smoothness_mean', 'value'),
     State('Input_compactness_mean', 'value'),
     State('Input_concavity_mean', 'value'),
     State('Input_concave_points_mean', 'value'),
     State('Input_symmetry_mean', 'value'),
     ]
)

def sein_predict(n_clicks, input_IDsein, input_texture_mean, input_area_mean,input_smoothness_mean,
                 input_compactness_mean, input_concavity_mean, input_concave_points_mean,input_symmetry_mean):
    if n_clicks == 0:
        raise PreventUpdate
    else:
        X_sein = df_sein[[ 'texture_mean', 'area_mean', 'smoothness_mean',
                          'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean']]
        y_sein = df_sein['diagnosis']
        X_train_sein, X_test_sein, y_train_sein, y_test_sein = train_test_split(X_sein, y_sein, random_state=42,
                                                                                train_size=0.75)
        X_train_scal_sein = scal_sein.fit_transform(X_train_sein)
        X_test_scal_sein = scal_sein.transform(X_test_sein)
        modelSein.fit(X_train_scal_sein, y_train_sein)
        sein_data = np.array(
            [input_texture_mean, input_area_mean, input_smoothness_mean,
             input_compactness_mean, input_concavity_mean, input_concave_points_mean, input_symmetry_mean,
             ]).reshape(1, X_sein.shape[1])
        sein_data = pd.DataFrame(sein_data, columns=X_sein.columns)
        sein_data_scal = scal_sein.transform(sein_data)
        sein_predict = modelSein.predict(sein_data_scal)
        current_date = datetime.datetime.now().strftime("%d-%m-%Y")
        sein_worksheet = gc.open('data_patient_sein').sheet1
        sein_worksheet.append_row([current_date, input_IDsein, input_texture_mean, input_area_mean,
                 input_smoothness_mean, input_compactness_mean, input_concavity_mean, input_concave_points_mean,
                 input_symmetry_mean, int(sein_predict)])
        sein_export = f"Données exportées : id {input_IDsein}"
        if sein_predict == 0:
            sein_diag = "C'est probablement une tumeur bénigne"
            proba = modelSein.predict_proba(sein_data_scal)
            sein_proba = proba[0,0]
            proba_diag_sein = f"Selon notre modèle, la fiabilité du résultat est évalué à  : {sein_proba*100} %"
            return sein_diag, proba_diag_sein, sein_export
        else:
            sein_diag = "C'est probablement une tumeur maligne"
            proba = modelSein.predict_proba(sein_data_scal)
            sein_proba = proba[0,1]
            proba_diag_sein = f"Selon notre modèle, la fiabilité du résultat est évalué à  : {sein_proba*100} %"
            return sein_diag, proba_diag_sein, sein_export

# Définir la mise en page globale de l'application
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(
        id='page-content',
        style={
            'padding-top': '70px',
            'background': 'linear-gradient(to top, #ffffff, #f0f8ff)',
            'height': '100vh',
            'width': '100%',
            'top': 0,
            'left': 0,
            'zIndex': -1
        }
    )
])

# Callback pour mettre à jour le contenu de la page en fonction de l'URL
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])

def display_page(pathname):
    if pathname == '/diabete':
        return diabete_layout
    elif pathname == '/maladie-chronique-renale':
        return mcr_layout
    elif pathname == '/foie':
        return foie_layout
    elif pathname == '/coeur':
        return coeur_layout
    elif pathname == '/sein':
        return sein_layout
    else:
        return accueil_layout


if __name__ == '__main__':
    app.run_server(debug=True)