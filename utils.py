import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import yaml

def read_config(config_path):
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        return config


def plotter(df, option, DURATION, EVENT, CategoricalDtype):
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(10,5))
    T = df[DURATION]
    E = df[EVENT]

    if isinstance(df[option].dtype, CategoricalDtype):
        unique_codes = list(set(df[option].cat.codes.values))
        unique_codes.sort()

        mapping = dict(zip(
            df[option].cat.codes.values,
            df[option].values))
        kmf.fit(T, E)

        for code in unique_codes:
            subset = (df[option] == mapping[code])
            kmf.fit(T[subset], event_observed=E[subset], label=mapping[code])
            kmf.plot_survival_function(ax=ax)

    else:
        unique_codes = list(set(df[option].values))
        unique_codes.sort()

        kmf.fit(T, E)

        for code in unique_codes:
            subset = (df[option] == code)
            kmf.fit(T[subset], event_observed=E[subset], label=code)
            kmf.plot_survival_function(ax=ax)


#    plt.title("Lifespans by " + option)
    return plt
