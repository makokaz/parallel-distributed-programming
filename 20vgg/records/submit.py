#!/usr/bin/python3

"""
submit execution log
"""

import argparse
import errno
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
import parse_log

dbg = 0

# --------- nuts and bolts ---------

def Ws(msg):
    """
    write to stdout
    """
    sys.stdout.write(msg)
    sys.stdout.flush()

def Es(msg):
    """
    write to stderr
    """
    sys.stderr.write(msg)

def chmod(filename, perm):
    """
    change mode
    """
    stat = os.stat(filename)
    if os.geteuid() == stat.st_uid:
        os.chmod(filename, perm)

def ensure_directory(directory):
    """
    ensure directory exists
    """
    try:
        os.makedirs(directory)
    except OSError as err:
        assert(err.errno == errno.EEXIST), err

# database nuts and bolts

def do_sql(con, cmd, dbg_level, *vals):
    """
    do sql statement.
    dbg_level is the level of verbosity above which
    it is printed
    """
    if dbg_level <= dbg:
        Es("%s with %s\n" % (cmd, vals))
    return con.execute(cmd, vals)

def read_schema(con):
    col_cmd = 'select name, sql from sqlite_master where type = "table"'
    pat = re.compile("CREATE TABLE [^\(]+\((?P<cols>.*)\)")
    schema = {}
    for name, create_table in do_sql(con, col_cmd, 1):
        m = pat.match(create_table)
        assert(m), create_table
        cols = m.group("cols")
        schema[name] = [s.strip() for s in cols.split(",")]
    return schema

def ensure_columns(con, schema, tbl, columns):
    """
    ensure table TBL exists and has COLUMNS
    """
    if tbl not in schema:
        create_cmd = "create table if not exists {}({})".format(tbl, ",".join(columns))
        do_sql(con, create_cmd, 1)
        schema[tbl] = columns
    else:
        existing_columns = schema[tbl]
        for col in columns:
            if col not in existing_columns:
                alt_cmd = "alter table {} add {}".format(tbl, col)
                do_sql(con, alt_cmd, 1)
                existing_columns.append(col)
        schema[tbl] = existing_columns

def open_for_transaction(sqlite3_file):
    """
    open database for transaction
    """
    con = sqlite3.connect(sqlite3_file)
    con.row_factory = sqlite3.Row
    schema = read_schema(con)
    return con, schema

def delete_from_db(con, schema, delete_seqids, user_me, delete_mine):
    """
    delete records of specified seqids from database
    """
    if "info" in schema:
        query_cmd = 'select distinct seqid from info where USER = ?'
        user_seqids = {row["seqid"] for row in do_sql(con, query_cmd, 1, user_me)}
    else:
        user_seqids = set()
    delete_seqids = set(delete_seqids)
    diff = delete_seqids.difference(user_seqids)
    if len(diff) > 0:
        Es("warning: you can't delete seqids %s, as they are not yours\n"
           % sorted(list(diff)))
    if delete_mine:
        seqids = user_seqids
    else:
        seqids = delete_seqids.intersection(user_seqids)
    if len(seqids) > 0:
        seqids_comma = ",".join([("%d" % x) for x in sorted(list(seqids))])
        for tbl, _ in schema.items():
            cmd = "delete from %s where seqid in (%s)" % (tbl, seqids_comma)
            do_sql(con, cmd, 1)
    return len(seqids)

def get_next_seqid(con, schema):
    """
    return next seqid
    """
    if "info" not in schema:
        return 0
    count_seqid_cmd = "select max(seqid) from info"
    [(seqid,)] = list(do_sql(con, count_seqid_cmd, 1))
    if seqid is None:
        return 0
    return seqid + 1

def parse_val(x):
    """
    parse a string into an sqlite3 value
    """
    if x is None:
        return None
    try:
        return int(x)
    except ValueError:
        pass
    try:
        return float(x)
    except ValueError:
        pass
    return x

def insert_row(con, schema, tbl, row, seqid):
    """
    insert a row into database
    """
    fields = ["seqid"] + list(row.keys())
    row["seqid"] = seqid
    n_fields = len(fields)
    ins_cmd = ("insert into {}({}) values({})"
               .format(tbl, ",".join(fields), ",".join(["?"] * n_fields)))
    vals = [row[f] for f in fields]
    ensure_columns(con, schema, tbl, fields)
    do_sql(con, ins_cmd, 2, *vals)
    return 1

def insert_rows(con, schema, tbl, rows, seqid):
    """
    insert rows into database
    """
    n_inserted = 0
    for row in rows:
        n_inserted += insert_row(con, schema, tbl, row, seqid)
    return n_inserted

def make_row_from_key_vals(rows):
    dic = {}
    keys = [row["key"] for row in rows]
    for row in rows:
        dic[row["key"]] = row["val"]
    return keys, dic

def insert_into_db(con, schema, logs):
    """
    insert all records in plogs into database
    """
    seqid = get_next_seqid(con, schema)
    n_inserted = 0
    for i, (parsed, _) in enumerate(logs):
        for tbl, rows in parsed.items():
            if tbl == "key_vals":
                _, dic = make_row_from_key_vals(rows)
                n_inserted += insert_row(con, schema, "info", dic, seqid + i)
            else:
                n_inserted += insert_rows(con, schema, tbl, rows, seqid + i)
    return n_inserted

def sqlite_to_json(con, schema, data_json):
    with open(data_json, "w") as wp:
        for tbl, cols in schema.items():
            cmd = "select {} from {}".format(",".join(cols), tbl)
            jsn = []
            for i, row in enumerate(do_sql(con, cmd, 1)):
                jsn.append(dict(row))
            wp.write("var {}_json = {};\n".format(tbl, json.dumps(jsn)))

def ensure_data_dir(data_dir):
    queue_dir = "{}/queue".format(data_dir)
    commit_dir = "{}/commit".format(data_dir)
    ensure_directory(data_dir)
    chmod(data_dir, 0o777)
    ensure_directory(queue_dir)
    chmod(queue_dir, 0o777)
    ensure_directory(commit_dir)
    chmod(commit_dir, 0o777)
    return queue_dir, commit_dir

# ------------------------------

def parse_delete_seqids(delete_seqids):
    """
    "1,2,3" --> [1,2,3]
    """
    if delete_seqids is None:
        return []
    ids = None
    try:
        ids = [int(x) for x in delete_seqids.split(",")]
    except ValueError:
        pass
    if ids is None:
        Es("argument to --delete-seqids must be N,N,...\n")
    return ids

def parse_logs(logs, q_dir):
    """
    parse all files in logs
    """
    result = []
    for log in logs:
        parsed, raw_data = parse_log.parse_log(log)
        if q_dir is None:
            q_log = None
        else:
            prefix = time.strftime("%Y-%m-%d-%H-%M-%S")
            fd, q_log = tempfile.mkstemp(suffix=".log", prefix=prefix, dir=q_dir)
            wp = os.fdopen(fd, "w")
            wp.write(raw_data)
            wp.close()
        result.append((parsed, q_log))
    return result

def move_to_dir(file, dir):
    orig_dir, orig_file = os.path.split(file)
    dest_file = "{}/{}".format(dir, orig_file)
    os.rename(file, dest_file)

def get_user():
    return os.environ.get("USER", "unknown")

def parse_args(argv):
    """
    parse command line args
    """
    psr = argparse.ArgumentParser()
    psr.add_argument("files", metavar="FILE",
                     nargs="*", help="files to submit")
    psr.add_argument("--dryrun", "-n",
                     action="store_true", help="dry run")
    psr.add_argument("--user", "-u", metavar="USER",
                     action="store", help="pretend user USER")
    psr.add_argument("--data", metavar="DIRECTORY",
                     default="viewer/dat", action="store",
                     help="generate database/csv under DIRECTORY")
    psr.add_argument("--delete-seqids", "-d", metavar="ID,ID,...",
                     action="store",
                     help="delete data of specified seqids")
    psr.add_argument("--delete-mine", "-D",
                     action="store_true",
                     help="delete all data of submitting user")
    opt = psr.parse_args(argv)
    if opt.user is None:
        opt.user = get_user()
    opt.delete_seqids = parse_delete_seqids(opt.delete_seqids)
    if opt.delete_seqids is None:
        return None
    return opt

def main():
    """
    main
    """
    args = parse_args(sys.argv[1:])
    if args is None:
        return 1
    logs = args.files[:]
    if args.dryrun:
        parse_logs(logs, None)
        Es("submit.py: dry run. do nothing\n")
        return 0
    data_dir = args.data
    q_dir, c_dir = ensure_data_dir(data_dir)
    # parse and queue the contents
    q_logs = parse_logs(logs, q_dir)
    a_sqlite = "{}/a.sqlite".format(data_dir)
    con, schema = open_for_transaction(a_sqlite)
    n_deleted = delete_from_db(con, schema,
                               args.delete_seqids, args.user, args.delete_mine)
    n_inserted = insert_into_db(con, schema, q_logs)
    con.commit()
    con.close()
    chmod(a_sqlite, 0o666)
    for _, q_log in q_logs:
        move_to_dir(q_log, c_dir)
    if n_deleted + n_inserted > 0:
        Es("database %s updated\n" % a_sqlite)
    return 0

main()
