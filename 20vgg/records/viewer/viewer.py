#!/usr/bin/python3
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import pandas as pd 
import numpy as np
import sqlite3

################################################
# the app object
################################################

app = dash.Dash(__name__)
a_sqlite = "dat/a.sqlite"

################################################
# nuts and bolts
################################################

def do_sql(conn, cmd):
    print(cmd)
    return conn.execute(cmd)

################################################
# selector + run table
################################################

def run_table_div():
    div = html.Div([
        html.P("sql selector"),
        dcc.Input(id="sql_selector"),
        html.Button("select", id="sql_selector_button"),
        dcc.Graph(id="run_table"),
        dcc.Graph(id="cols_table"),
    ])
    return div

@app.callback(
    Output("run_table",    "figure"),
    Output("cols_table",   "figure"),
    Input( "sql_selector_button", "n_clicks"),
    State( "sql_selector", "value"),
)
def update_run_table(n_clicks, cond):
    conn = sqlite3.connect(a_sqlite)
    conn.row_factory = sqlite3.Row
    where = "where {}".format(cond) if cond else ""
    cols = ["seqid", "USER", "host", "algo_s", "algo", "gpu_algo", 
            "batch_sz", "learnrate", "start_at", "end_at", 
            "SLURM_JOB_CPUS_PER_NODE", "SLURM_NTASKS", "SLURM_NPROCS"]
    cmd = ("select {} from info {} order by seqid"
           .format(",".join(cols), where))
    result = list(do_sql(conn, cmd))
    conn.close()
    if len(result) > 0:
        row = result[0]
        cols = list(row.keys())
        cells = [[row[c] for row in result] for c in cols]
    else:
        cols = []
        cells = []
    table = go.Table(header=dict(values=cols), cells=dict(values=cells))
    run_tbl = go.Figure(data=[table])
    run_tbl.update_layout(height=1000)
    return run_tbl, run_tbl

################################################
# loss accuracy
################################################

def loss_accuracy_graph_div():
    cols = ["samples", "t",
            "train_loss", "train_accuracy", "validate_loss", "validate_accuracy"]
    options = [{'label': x, 'value': x} for x in cols]
    div = html.Div([
        html.P("x:"),
        dcc.RadioItems(id="loss_accuracy_graph_x", options=options, value="samples"),
        html.P("y:"),
        dcc.RadioItems(id="loss_accuracy_graph_y", options=options, value="train_loss"),
        dcc.Graph(id="loss_accuracy_graph"),
    ])
    return div

@app.callback(
    Output("loss_accuracy_graph",      "figure"),
    Input( "loss_accuracy_graph_x",    "value"),
    Input( "loss_accuracy_graph_y",    "value"),
    Input( "sql_selector_button", "n_clicks"),
    State( "sql_selector", "value"),
)
def update_loss_accuracy_graph(selected_x, selected_y, n_clicks, cond):
    conn = sqlite3.connect(a_sqlite)
    where = "where {}".format(cond) if cond else ""
    cmd = "select seqid from info {}".format(where)
    seqids = [str(seqid) for seqid, in do_sql(conn, cmd)]
    cmdx = ("select {},{},seqid from loss_accuracy where seqid in ({}) order by {}"
            .format(selected_x, selected_y, ",".join(seqids), selected_x))
    result = list(do_sql(conn, cmdx))
    conn.close()
    x = [x for x,_,_ in result]
    y = [y for _,y,_ in result]
    seqid = [seqid for _,_,seqid in result]
    df = pd.DataFrame(dict(x=x, y=y, seqid=seqid))
    fig = px.line(df, x="x", y="y", color="seqid")
    return fig

################################################
# kernel times table
################################################

def kernel_times_table_div():
    div = html.Div([
        dcc.Graph(id="kernel_times_table"),
    ])
    return div

#@app.callback(
#    Output("kernel_times_table",      "figure"),
#    Input( "sql_selector", "value"),
#)
def update_kernel_times_table(kernel_times_table_cond):
    conn = sqlite3.connect(a_sqlite)
    where = "where {}".format(kernel_times_table_cond) if kernel_times_table_cond else ""
    cols = ["seqid", "cls", "cargs", "fun", "fargs", "sum(t1-t0)", "sum(dt)"]
    cmd = ("""select {} from kernel_times 
    {}
    group by seqid,cls,cargs,fun,fargs"""
           .format(",".join(cols), where))
    result = list(do_sql(conn, cmd))
    conn.close()
    cells = [[row[i] for row in result] for i in range(len(cols))]
    table = go.Table(header=dict(values=cols), cells=dict(values=cells))
    fig = go.Figure(data=[table])
    fig.update_layout(height=1000)
    return fig

################################################
# kernel times graph
################################################

def kernel_times_bar_chart_div():
    div = html.Div([
        html.P("group by"),
        dcc.Input(id="kernel_times_bar_chart_group_by", value="cls,fun"),
        dcc.Graph(id="kernel_times_bar_chart"),
    ])
    return div

def make_kernel_name(row, group_by):
    dic = dict(row)
    cls = dic.get("cls")
    cargs = dic.get("cargs")
    fun = dic.get("fun")
    fargs = dic.get("fargs")
    cls_cargs = "{}{}".format((cls if cls else ""), (cargs if cargs else ""))
    fun_fargs = "{}{}".format((fun if fun else ""), (fargs if fargs else ""))
    if cls_cargs:
        return "{}::{}".format(cls_cargs, fun_fargs)
    else:
        return fun_fargs

@app.callback(
    Output("kernel_times_bar_chart",      "figure"),
    Input( "sql_selector_button", "n_clicks"),
    State( "sql_selector", "value"),
    State( "kernel_times_bar_chart_group_by", "value"),
)
def update_kernel_times_bar_chart(n_clicks, cond, group_by):
    conn = sqlite3.connect(a_sqlite)
    conn.row_factory = sqlite3.Row
    where = "where {}".format(cond) if cond else ""
    cmd = "select seqid from info {}".format(where)
    seqids = [str(seqid) for seqid, in do_sql(conn, cmd)]
    fields = "seqid,{}".format(group_by) if group_by else "seqid"
    cmd = ("""select {},sum(dt)
    from kernel_times 
    where seqid in ({})
    group by {}
    order by seqid,sum(dt) desc
"""
           .format(fields, ",".join(seqids), fields))
    result = list(do_sql(conn, cmd))
    conn.close()
    seqid = [row["seqid"] for row in result]
    kernel = [make_kernel_name(row, fields) for row in result]
    sum_dt = [row["sum(dt)"] for row in result]
    df = pd.DataFrame({"seqid" : seqid, "kernel" : kernel, "sum_dt" : sum_dt})
    fig = px.bar(df, x="seqid", y="sum_dt", color="kernel")
    fig.update_layout(height=2000)
    return fig

################################################
# the whole page
################################################

app.layout = html.Div(
    [
        html.H2("select"),
        run_table_div(),
        html.H2("loss/accuracy over samples/time"),
        loss_accuracy_graph_div(),
        html.H2("kernel times bar chart"),
        kernel_times_bar_chart_div(),
    ],
    style={"padding": "2%", "margin": "auto"},
)

if __name__ == "__main__":
    app.run_server(debug=True)
