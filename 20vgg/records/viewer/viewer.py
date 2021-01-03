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

if __name__ == "__main__":
    app = dash.Dash(__name__)
else:
    app = dash.Dash(__name__, requests_pathname_prefix='/viewer/')

application = app.server
a_sqlite = "/home/tau/public_html/lecture/parallel_distributed/parallel-distributed-handson/20vgg/records/vgg_records/a.sqlite"

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
        html.P("Enter a condition (SQL expression) to filter runs you want to display. e.g.,"),
        html.Ul([
            html.Li("seqid = 1"),
            html.Li('host like "big%"'),
            html.Li('USER = "u01234"'),
            html.Li('USER = "u01234" and algo_s like "cpu_%"'),
        ]),
        html.P(('In order to be useful, you do not want to display too many runs.'
                ' Come up with a filtering expression that chooses what you want to compare.'
                ' I will hopefully make some buttons to quickly display most "interesting" runs'
                ' (e.g., "best" in various criterion, such as best samples/time, best achieved loss, etc.)')),
        dcc.Input(id="sql_selector"),
        html.Button("select", id="sql_selector_button"),
        html.P("? runs selected", id="how_many_runs"),
        dcc.Graph(id="run_table"),
        # dcc.Graph(id="cols_table"),
    ])
    return div

@app.callback(
    Output("how_many_runs",  "children"),
    Output("run_table",      "figure"),
    # Output("cols_table",   "figure"),
    Input( "sql_selector_button", "n_clicks"),
    State( "sql_selector", "value"),
)
def update_run_table(n_clicks, cond):
    conn = sqlite3.connect(a_sqlite)
    conn.row_factory = sqlite3.Row
    where = "where {}".format(cond) if cond else ""
    cols = ["seqid", "USER", "owner", "host", "algo_s", "gpu_algo", 
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
    run_tbl.update_layout(height=800)
    n_runs = "%d runs selected" % len(result)
    #return run_tbl, run_tbl
    return n_runs, run_tbl

################################################
# loss accuracy
################################################

def loss_accuracy_graph_div():
    cols = ["samples", "t",
            "train_loss", "train_accuracy", "validate_loss", "validate_accuracy"]
    options = [{'label': x, 'value': x} for x in cols]
    div = html.Div([
        html.P("This section is mainly for displaying how loss or accuracy evolves over time.  x-axis is typically t (for wall clock time time) or samples (the number of samples trained) and y-axis loss (measured by the cross entropy between the predicted probability distribution over the ten classses and the true distribution (1 for the true class and 0 for others)) or accuracy (the proportion of the correctly labeled samples).  Each is measured for training samples (a mini batch) or the samples left for validation."),
        html.P("You may also want to set x-axis to t and y-axis samples, to show the throughput of your program in terms of samples/sec."),
        html.P("choose x-axis:"),
        dcc.RadioItems(id="loss_accuracy_graph_x", options=options, value="samples"),
        html.P("choose y-axis:"),
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
        html.P("Per-sample execution time of each kernel, in the form of stacked bar chart."
               " This is useful when you want to see the performance of your parallelized/vectorized/optimized code relative to baseline (either by you or friends)."
               " Each area in the stack shows time spent per sample in each function."
               " That is, the time is the total time spent in each function / total number of samples processed."),
        html.P("The expression below specifies attributes with which to group (aggregate) execution times."
               "You can specify comma-separated list of the following."),
        html.Ul([
            html.Li("cls : kernel name such as Convolution2D, Linear, etc.  Each kernel is implemented as a class template taking size parameters."),
            html.Li("cargs : values that instantiate the class templates, representing the size of input/output/parameters"),
            html.Li("fun : function name, either of forward, backward, or update"),
        ]
        ),
        html.P("For example, cls,fun distinguishes Convolution2D::forward, Convolution2D::backward, Convolution2D::update, Linear::forward, etc.,"
               " but does not distinguish different instantiations of Convolution2D with different size parameters."),
        html.Ul([
            html.Li(["select: ", dcc.Input(id="kernel_times_bar_chart_cond", value="") ]),
            html.Li(["group by: ", dcc.Input(id="kernel_times_bar_chart_group_by", value="cls,fun")])
        ]),
        html.Button("update", id="kernel_times_bar_chart_update_button"),
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
    Input( "kernel_times_bar_chart_update_button", "n_clicks"),
    State( "sql_selector", "value"),
    State( "kernel_times_bar_chart_cond", "value"),
    State( "kernel_times_bar_chart_group_by", "value"),
)
def update_kernel_times_bar_chart(sql_selector_n_clicks, kernel_times_n_clicks, cond0, cond1, group_by):
    conn = sqlite3.connect(a_sqlite)
    conn.row_factory = sqlite3.Row
    where0 = "where {}".format(cond0) if cond0 else ""
    cmd = "select seqid from info {}".format(where0)
    seqids = [str(seqid) for seqid, in do_sql(conn, cmd)]
    fields = "seqid,{}".format(group_by) if group_by else "seqid"
    where1 = "and {}".format(cond1) if cond1 else ""
    cmd = ("""select {},sum(dt)/sum(b-a) as avg_dt
    from kernel_times 
    where seqid in ({}) {}
    group by {}
    order by seqid,avg_dt desc
"""
           .format(fields, ",".join(seqids), where1, fields))
    result = list(do_sql(conn, cmd))
    conn.close()
    seqid = [row["seqid"] for row in result]
    kernel = [make_kernel_name(row, fields) for row in result]
    avg_dt = [row["avg_dt"] for row in result]
    df = pd.DataFrame({"seqid" : seqid, "kernel" : kernel, "avg_dt" : avg_dt})
    fig = px.bar(df, x="seqid", y="avg_dt", color="kernel")
    fig.update_layout(height=2000)
    return fig

################################################
# the whole page
################################################

app.layout = html.Div(
    [
        html.H2("Select runs to display"),
        run_table_div(),
        html.H2("Loss/accuracy evolution with samples/time"),
        loss_accuracy_graph_div(),
        html.H2("Execution time breakdown"),
        kernel_times_bar_chart_div(),
    ],
    style={"padding": "2%", "margin": "auto"},
)

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0")
