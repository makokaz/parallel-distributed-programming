#!/usr/bin/python
import sys,re,time,socket,os,csv,sqlite3,errno,argparse

default_data_dir = "/home/u000/public_html/lecture/parallel_distributed/2016/examples/18mnist/mnist_records/public_data"
# default_log_dir = "mnist_log"

def Ws(s):
    sys.stdout.write(s)

def Es(s):
    sys.stderr.write(s)

def ensure_directory(directory):
    try:
        os.makedirs(directory)
    except OSError,e:
        assert(e.errno == errno.EEXIST), e
    

schema = [
    # [ "series", "seqid,sample_start,sample_end,clocks,t,correct,samples,accuracy,cross_entropy".split(",") ],
    [ "series", "seqid,sample_start,sample_end,clocks,t,correct,cross_entropy".split(",") ],
    [ "attr",   "seqid,key,val".split(",") ]
]

def parse_an_execution(fp):
    """
    parse a single execution record
    returns { table -> list of rows }
    """
    exec_rows = {}
    for i,(tbl,_) in enumerate(schema):
        rp = csv.DictReader(fp)
        if rp.fieldnames is None:
            # true EOF
            assert(i == 0)
            return None
        exec_rows[tbl] = []
        for row in rp:
            if all(x == "" for x in row.values()):
                break
            exec_rows[tbl].append(row)
    return exec_rows

def parse_executions(files):
    exec_records = []
    for f in files:
        if f == "-":
            fp = sys.stdin
        else:
            fp = open(f, "rb")
        while 1:
            record = parse_an_execution(fp)
            if record is None: break
            exec_records.append(record)
        if f != "-":
            fp.close()
    return exec_records

def timestamp():
    return int(time.time() * 1000)
        
def parse_val(x):
    try:
        return int(x)
    except ValueError,e:
        pass
    try:
        return float(x)
    except ValueError,e:
        pass
    return x

def insert_row(con, tbl, row, seqid):
    row["seqid"] = seqid
    fields = row.keys()
    nf = len(fields)
    ins_cmd = ("insert into %s(%s) values(%s)"
               % (tbl, ",".join(fields), ",".join(["?"] * nf)))
    con.execute(ins_cmd, tuple([ parse_val(row[f]) for f in fields ]))

def insert_rows(con, tbl, rows, seqid):
    for row in rows:
        insert_row(con, tbl, row, seqid)

def open_for_transaction(sqlite3_file):
    con = sqlite3.connect(sqlite3_file)
    con.row_factory = sqlite3.Row
    for tbl,fields in schema:
        create_cmd = ("create table if not exists %s(%s)"
                      % (tbl, ",".join(fields)))
        con.execute(create_cmd)
    return con

def get_next_seqid(con):
    count_seqid_cmd = "select max(seqid) from series"
    [ (seqid,) ] = con.execute(count_seqid_cmd).fetchall()
    if seqid is None:
        return 0
    else:
        return seqid + 1
        
def insert_into_db(con, all_execs, additional_attr):
    seqid = get_next_seqid(con)
    n_inserted = 0
    for i,exec_record in enumerate(all_execs):
        for tbl,rows in exec_record.items():
            insert_rows(con, tbl, rows, seqid + i)
            n_inserted += len(rows)
        for k,v in additional_attr.items():
            insert_row(con, "attr", { "key" : k, "val" : v }, seqid + i)
            n_inserted += 1
    return n_inserted

def delete_from_db(con, delete_seqids, me, delete_mine):
    query_cmd = 'select seqid from attr where key = "user" and val = "%s"' % me
    user_seqids = set([ row["seqid"] for row in con.execute(query_cmd).fetchall() ])
    delete_seqids = set(delete_seqids)
    diff = delete_seqids.difference(user_seqids)
    if len(diff) > 0:
        Es("warning: you can't delete seqids %s, as they are not yours\n" % sorted(list(diff)))
    if delete_mine:
        seqids = user_seqids
    else:
        seqids = delete_seqids.intersection(user_seqids)
    if len(seqids) > 0:
        X = ",".join([ ("%d" % x) for x in sorted(list(seqids)) ])
        for tbl in [ "series", "attr" ]:
            cmd = "delete from %s where seqid in (%s)" % (tbl, X)
            con.execute(cmd)
    return len(seqids)

def chmod(filename, perm):
    s = os.stat(filename)
    if os.geteuid() == s.st_uid:
        os.chmod(filename, perm)

def gen_csv(con, csv_file, table):
    wp = open(csv_file, "wb")
    csv_wp = None
    for row in con.execute("select * from %s" % table):
        if csv_wp is None:
            csv_wp = csv.DictWriter(wp, fieldnames=row.keys())
            csv_wp.writeheader()
        csv_wp.writerow(dict(row))
    wp.close()
    Es("csv written to %s\n" % csv_file)
    chmod(csv_file, 0666)
        
def get_user(user):
    if user is not None:
        return user
    else:
        return os.environ.get("USER", "unknown")

def parse_args(argv):
    ps = argparse.ArgumentParser()
    ps.add_argument("files", metavar="FILE",
                    nargs="*", help="files to submit")
    ps.add_argument("--dryrun", "-n",
                    action="store_true", help="dry run")
    ps.add_argument("--user", "-u", metavar="USER",
                    action="store", help="pretend user USER")
    ps.add_argument("--data", metavar="DIRECTORY",
                    action="store",
                    help="generate database/csv under DIRECTORY")
    ps.add_argument("--delete-seqids", "-d", metavar="ID,ID,...",
                    action="store",
                    help="delete data of specified seqids")
    ps.add_argument("--delete-mine", "-D", 
                    action="store_true",
                    help="delete all data of submitting user")
    return ps.parse_args(argv)

def parse_delete_seqids(delete_seqids):
    if delete_seqids is None:
        return []
    else:
        ids = None
        try:
            ids = [ int(x) for x in delete_seqids.split(",") ]
        except ValueError,e:
            pass
        if ids is None:
            Es("argument to --delete-seqids must be N,N,...\n")
        return ids
    
def main():
    args = parse_args(sys.argv[1:])
    delete_seqids = parse_delete_seqids(args.delete_seqids)
    if delete_seqids is None:
        sys.exit(1)
    data_dir = args.data if args.data else default_data_dir
    sqlite3_file = "%s/a.sqlite"   % data_dir
    series_csv   = "%s/series.csv" % data_dir
    attr_csv     = "%s/attr.csv"   % data_dir
    files = args.files[:]
    if len(delete_seqids) == 0 and (not args.delete_mine) and len(files) == 0:
        files.append("-")
    exec_records = parse_executions(files)
    if args.dryrun:
        Es("submit.py: dry run. do nothing\n")
    else:
        user = get_user(args.user)
        add_attr = { "submit_time" : timestamp(),
                     "user" : user }
        ensure_directory(data_dir)
        chmod(data_dir, 0777)
        con = open_for_transaction(sqlite3_file)
        n_deleted = delete_from_db(con, delete_seqids, user, args.delete_mine)
        n_inserted = insert_into_db(con, exec_records, add_attr)
        con.commit()
        if n_deleted + n_inserted > 0:
            Es("database %s updated\n" % sqlite3_file)
            gen_csv(con, series_csv, "series")
            gen_csv(con, attr_csv, "attr")
        con.close()
        chmod(sqlite3_file, 0666)
        Es("check http://parallel.hopto.org/~u000/lecture/parallel_distributed/2016/examples/18mnist/mnist_records/\n")

main()
