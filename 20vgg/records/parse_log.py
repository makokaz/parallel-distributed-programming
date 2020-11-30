#!/usr/bin/python3
import csv
import json
import re
import sys
import pdb

class parse_error(Exception):
    pass

class log_parser_base:
    def __init__(self, fp):
        self.fp = fp
        tokens = {
            "open_log"          : "open a log (?P<when>.+)",
            "close_log"         : "close a log (?P<when>.+)",
            "env"               : "(?P<var>[A-Za-z0-9_]+)( undefined|=(?P<val>.+))",
            "model_start"       : "model building starts",
            "model_end"         : "model building ends",
            "load_start"        : "loading (?P<n_training>\d+)/(?P<n_validation>\d+) training/validation data from (?P<data>.+) starts",
            "load_end"          : "loading data ends",
            "train_data"        : "train: (?P<data>.+)",
            "validation_data"   : "validate: (?P<data>.+)",
            "training_start"    : "training starts",
            "training_end"      : "training ends",
            "train_begin"       : "=== train (?P<a>\d+) - (?P<b>\d+) ===",
            "validate_begin"    : "=== validate (?P<a>\d+) - (?P<b>\d+) ===",
            "sample"            : "sample (?P<sample>\d+) image (?P<image>\d+) pred (?P<pred>\d+) truth (?P<truth>\d+)",
            "train_accuracy"    : "train accuracy (?P<correct>\d+) / (?P<batch_sz>\d+) = (?P<accuracy>\d+\.\d+)",
            "validate_accuracy" : "validate accuracy (?P<correct>\d+) / (?P<batch_sz>\d+) = (?P<accuracy>\d+\.\d+)",
            "train_loss"        : "train loss = (?P<loss>\d+\.\d+)",
            "validate_loss"     : "validate loss = (?P<loss>\d+\.\d+)",
            "kernel_start"      : "(?P<kernel>.*): starts",
            "kernel_end"        : "(?P<kernel>.*): ends\. took (?P<kernel_time>\d+) nsec",
        }
        pat_time = "(?P<t>\d+): "
        self.patterns = {tok : re.compile(pat_time + pat) for tok, pat in tokens.items()}
        self.patterns["EOF"] = re.compile("^$")
        self.line_no = 0
        self.next_line()

    def next_line(self):
        """
        get next line, set token kind
        """
        self.line = self.fp.readline()
        self.line_no += 1
        for tok, regex in self.patterns.items():
            m = regex.match(self.line)
            if m:
                self.tok = tok
                self.data = m.groupdict()
                return
        self.tok = None
        self.data = None
    def call_action(self, tok):
        method_name = "action_%s" % tok
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            method(self.data)
    def eat(self, tok):
        if self.tok != tok:
            self.parse_error(tok)
        self.call_action(tok)
        self.next_line()
    def parse_error(self, tok):
        raise parse_error("%s:%d:error: expected %s but got %s\n[%s]\n" %
                          (self.fp.name, self.line_no, tok, self.tok, self.line))
    def parse_kernel_start(self):
        self.kpsr.parse(self.data["kernel"])
        self.eat("kernel_start")
    def parse_kernel_end(self):
        self.kpsr.parse(self.data["kernel"])
        self.eat("kernel_end")
    def parse_mini_batch(self):
        while self.tok == "kernel_start":
            self.parse_kernel_start()
            self.parse_kernel_end()
        while self.tok == "sample":
            self.eat("sample")
    def parse_train(self):
        """
        === train 448 - 512 ===
        ...
        """
        self.eat("train_begin")
        self.parse_mini_batch()
        self.eat("train_accuracy")
        self.eat("train_loss")
    def parse_validate(self):
        """
        === validate 448 - 512 ===
        ...
        """
        self.eat("validate_begin")
        while self.tok == None:
            self.parse_mini_batch()
        self.eat("validate_accuracy")
        self.eat("validate_loss")
    def parse_file(self):
        """
        other* train_or_validate* EOF
        """
        self.eat("open_log")
        while self.tok == "env":
            self.eat("env")
        self.eat("model_start")
        self.eat("model_end")
        self.eat("load_start")
        self.eat("train_data")
        self.eat("validation_data")
        self.eat("load_end")
        self.eat("training_start")
        while self.tok in ["train_begin", "validate_begin"]:
            if self.tok == "train_begin":
                self.parse_train()
            elif self.tok == "validate_begin":
                self.parse_validate()
            else:
                assert(self.tok in ["train_begin", "validate_begin"]), self.tok
        self.eat("training_end")
        self.eat("close_log")
        self.eat("EOF")

class kernel_parser:
    def __init__(self):
        self.kw = ["with"]
        self.patterns_ = [
            ("id", "[A-Za-z_][0-9A-Za-z_]*"),
            ("num", "[1-9][0-9]*"),
            ("<" ,  "<"),
            ("," ,  ","),
            (">" ,  ">"),
            ("&" ,  "&"),
            ("::",  "::"),
            ("(" ,  "\("),
            (")" ,  "\)"),
            ("[" ,  "\["),
            ("]" ,  "\]"),
            ("=" ,  "="),
            ("+" ,  "\+"),
            ("-" ,  "\-"),
            ("*" ,  "\*"),
            ("/" ,  "\/"),
            ("%" ,  "%"),
            (";" ,  ";"),
        ]
        self.patterns = {(k, re.compile(v)) for k, v in self.patterns_}
        self.tokenize_pattern = re.compile("(%s)" % "|".join([v for k, v in self.patterns_]))
        self.first_type = set(["id"])
    def init(self, s):
        self.tokens = self.tokenize_pattern.findall(s)
        self.idx = -1
        self.next_token()
    def next_token(self):
        self.idx += 1
        if self.idx < len(self.tokens):
            self.tok = self.tokens[self.idx]
            self.kind = self.token_kind(self.tok)
        else:
            self.tok = ""
            self.kind = "EOF"
        return self.kind
    def token_kind(self, tok):
        for k, p in self.patterns:
            if p.match(tok):
                if tok in self.kw:
                    return tok
                return k
        assert(0), tok
    def eat(self, kinds):
        assert(isinstance(kinds, type([]))), kinds
        for k in kinds:
            if self.kind == k:
                tok = self.tok
                self.next_token()
                return tok
        self.parse_error(kinds)
    def parse_error(self, kinds):
        raise parse_error("error: expected %s but got '%s' (%s)"
                          % (kinds, self.kind, self.tok))
    def parse_template_expr(self):
        """
        expr ::= multiplicative ( +/- expr )*
        multiplicative ::= primary (*// multiplicative)*
        primary ::= id | num | ( expr )
        """
        expr = self.parse_multiplicative()
        while self.kind in [ "+", "-" ]:
            op = self.eat([self.kind])
            expr = (op, expr, self.parse_template_expr())
        return expr
    def parse_multiplicative(self):
        """
        multiplicative ::= primary (*// multiplicative)*
        primary ::= id | num | ( expr )
        """
        expr = self.parse_primary()
        while self.kind in [ "*", "/" ]:
            op = self.eat([self.kind])
            expr = (op, expr, self.parse_multiplicative())
        return expr
    def parse_primary(self):
        """
        primary ::= id | num | ( expr )
        """
        if self.kind == "(":
            self.eat(["("])
            expr = self.parse_template_expr()
            self.eat([")"])
            return expr
        elif self.kind == "id":
            return self.eat(["id"])
        elif self.kind == "num":
            return int(self.eat(["num"]))
        else:
            raise parse_error("primary expected (, id, or num, got %s"
                              % self.kind)
    def parse_id(self):
        """
        var<expr, expr, ...>
        """
        name = self.eat(["id"])
        if self.kind == "<":
            args = []
            self.eat(["<"])
            if self.kind == "id":
                args.append(self.parse_template_expr())
                while self.kind == ",":
                    self.eat([","])
                    args.append(self.parse_template_expr())
            self.eat([">"])
        else:
            args = None
        return dict(name=name, args=args)
    def parse_type(self):
        """
        var<expr, expr, ...>[&]
        """
        d = self.parse_id()
        name = d["name"]
        args = d["args"]
        if self.kind == "&":
            amp = self.eat(["&"])
        else:
            amp = None
        return dict(name=name, args=args, amp=amp)
    def parse_class_fun_name(self):
        """
        var<expr, expr, ...>::var
        """
        d = self.parse_id()
        if self.kind == "::":
            self.eat(["::"])
            class_name = d["name"]
            class_args = d["args"]
            f = self.parse_id()
            fun_name = f["name"]
            fun_args = f["args"]
        else:
            class_name = None
            class_args = None
            fun_name = d["name"]
            fun_args = d["args"]
        return dict(class_name=class_name, class_args=class_args,
                    fun_name=fun_name, fun_args=fun_args)
    def parse_instantiation(self):
        """
        type id = expr | id = type_expr
        """
        n0 = self.eat(["id"])
        if self.kind == "id":
            # int x  = 10
            typ = n0
            var = self.eat(["id"])
            self.eat(["="])
            expr = self.parse_template_expr()
            return (var, ("val", expr))
        elif self.kind == "=":
            # e.g., real = float
            typ = None
            var = n0
            self.eat(["="])
            expr = self.parse_type()
            return (var, ("type", expr))
        else:
            raise parse_error("instantiation")
    def parse_kernel_sig(self):
        """
        type fun(type,type,..)
        """
        return_type = self.parse_type()
        class_fun = self.parse_class_fun_name()
        self.eat(["("])
        params = []
        if self.kind in self.first_type:
            params.append(self.parse_type())
            while self.kind == ",":
                self.eat([","])
                params.append(self.parse_type())
        self.eat([")"])
        instantiations = {}
        if self.kind == "[":
            self.eat(["["])
            self.eat(["with"])
            if self.kind in self.first_type:
                var, val = self.parse_instantiation()
                instantiations[var] = val
                while self.kind == ";":
                    self.eat([";"])
                    var, val = self.parse_instantiation()
                    instantiations[var] = val
            self.eat(["]"])
        self.eat(["EOF"])
        return dict(return_type=return_type, class_fun=class_fun,
                    params=params, instantiations=instantiations)
    def parse(self, s):
        self.init(s)
        return self.parse_kernel_sig()

class log_parser(log_parser_base):
    def __init__(self, fp):
        super().__init__(fp)
        self.samples = []
        self.n_training_samples = 0
        self.loss_acc = []
        self.kpsr = kernel_parser()
        self.kernels = []
    def action_train_begin(self, data):
        self.n_training_samples = int(data["b"])
        self.samples.append(("t", []))
    def action_validate_begin(self, data):
        self.samples.append(("v", []))
    def action_train_loss(self, data):
        self.loss_acc.append((self.n_training_samples, "train_loss",        int(data["t"]), float(data["loss"])))
    def action_validate_loss(self, data):
        self.loss_acc.append((self.n_training_samples, "validate_loss",     int(data["t"]), float(data["loss"])))
    def action_train_accuracy(self, data):
        self.loss_acc.append((self.n_training_samples, "train_accuracy",    int(data["t"]), float(data["accuracy"])))
    def action_validate_accuracy(self, data):
        self.loss_acc.append((self.n_training_samples, "validate_accuracy", int(data["t"]), float(data["accuracy"])))
    def action_sample(self, data):
        t_or_v, cur = self.samples[-1]
        cur.append(data)
    def action_kernel_start(self, data):
        kernel = self.kpsr.parse(data["kernel"])
        self.kernels.append((int(data["t"]), None, kernel, None))
    def action_kernel_end(self, data):
        kernel = self.kpsr.parse(data["kernel"])
        kernel_time = int(data["kernel_time"])
        k1 = self.kernels[-1]
        t0, t1, ks, kt = k1
        assert(t1 is None), k1
        assert(ks == kernel), k1
        assert(kt is None), k1
        t1 = int(data["t"])
        kt = kernel_time
        self.kernels[-1] = (t0, t1, ks, kt)
    def write_samples_csv(self, filename):
        with open(filename, "w") as wp:
            csv_wp = csv.DictWriter(wp, ["iter", "t_v", "image", "pred", "truth", "sample", "t"])
            csv_wp.writeheader()
            for i, (t_or_v, samples) in enumerate(self.samples):
                for s in samples:
                    csv_wp.writerow(dict(iter=i, t_v=t_or_v, **s))
    def write_samples_json(self):
        js = []
        for i, (t_or_v, samples) in enumerate(self.samples):
            for s in samples:
                js.append(dict(iter=i, t_v=t_or_v, **s))
        return js
    def eval_template_arg(self, expr, env):
        if isinstance(expr, type("")):
            # variable
            kind, val = env[expr]
            return val
        elif isinstance(expr, type(0)):
            # number
            return expr
        elif isinstance(expr, type(())):
            # (op, e0, e1)
            op, e0, e1 = expr
            v0 = self.eval_template_arg(e0, env)
            v1 = self.eval_template_arg(e0, env)
            if op == "+":
                return v0 + v1
            elif op == "-":
                return v0 - v1
            elif op == "*":
                return v0 * v1
            elif op == "/":
                return v0 / v1
            else:
                assert(op in ["+", "-", "*", "/", "%"]), op
    def instantiate(self, kernel):
        rt = kernel["return_type"]
        class_fun = kernel["class_fun"]
        params = kernel["params"]
        insts = kernel["instantiations"]
        class_name = class_fun["class_name"]
        class_args = class_fun["class_args"]
        fun_name = class_fun["fun_name"]
        fun_args = class_fun["fun_args"]
        if class_args is None:
            class_arg_vals = None
        else:
            class_arg_vals = [self.eval_template_arg(arg, insts) for arg in class_args]
        if fun_args is None:
            fun_arg_vals = None
        else:
            fun_arg_vals = [self.eval_template_arg(arg, insts) for arg in fun_args]
        return (class_name, class_arg_vals, fun_name, fun_arg_vals)
    def write_kernel_times_csv(self, filename):
        with open(filename, "w") as wp:
            csv_wp = csv.DictWriter(wp, ["t0", "t1", "cls", "cargs", "fun", "fargs", "dt"])
            csv_wp.writeheader()
            for t0, t1, kernel, dt in self.kernels:
                cls, cargs, fun, fargs = self.instantiate(kernel)
                if cargs is not None:
                    cargs = "<%s>" % ",".join("%s" % x for x in cargs)
                if fargs is not None:
                    fargs = "<%s>" % ",".join("%s" % x for x in fargs)
                csv_wp.writerow(dict(t0=t0, t1=t1, cls=cls, cargs=cargs, fun=fun, fargs=fargs, dt=dt))
    def write_kernel_times_json(self):
        js = []
        for t0, t1, kernel, dt in self.kernels:
            cls, cargs, fun, fargs = self.instantiate(kernel)
            if cargs is not None:
                cargs = "<%s>" % ",".join("%s" % x for x in cargs)
            if fargs is not None:
                fargs = "<%s>" % ",".join("%s" % x for x in fargs)
            js.append(dict(t0=t0, t1=t1, cls=cls, cargs=cargs, fun=fun, fargs=fargs, dt=dt))
        return js
    def write_loss_accuracy_csv(self, filename):
        with open(filename, "w") as wp:
            csv_wp = csv.DictWriter(wp, ["samples", "t",
                                         "train_accuracy", "validate_accuracy",
                                         "train_loss", "validate_loss"])
            csv_wp.writeheader()
            data = None
            for samples, kind, t, x in self.loss_acc:
                if data is None or data["samples"] != samples:
                    if data is not None:
                        csv_wp.writerow(data)
                    data = {"samples" : samples,
                            "t" : "",
                            "train_accuracy" : "",
                            "validate_accuracy" : "",
                            "train_loss" : "",
                            "validate_loss" : ""}
                data[kind] = x
                if kind == "train_accuracy":
                    data["t"] = t
    def write_loss_accuracy_json(self):
        data = None
        js = []
        for samples, kind, t, x in self.loss_acc:
            if data is None or data["samples"] != samples:
                if data is not None:
                    js.append(data)
                data = {"samples" : samples,
                        "t" : "",
                        "train_accuracy" : "",
                        "validate_accuracy" : "",
                        "train_loss" : "",
                        "validate_loss" : ""}
            data[kind] = x
            if kind == "train_accuracy":
                data["t"] = t
        return js
            
def main():
    log = sys.argv[1] if len(sys.argv) > 1 else "../vgg.log"
    with open(log) as fp:
        psr = log_parser(fp)
        psr.parse_file()
        if 0:
            psr.write_samples_csv("samples.csv")
            psr.write_loss_accuracy_csv("loss_accuracy.csv")
            psr.write_kernel_times_csv("kernel_times.csv")
        samples_json = psr.write_samples_json()
        loss_accuracy_json = psr.write_loss_accuracy_json()
        kernel_times_json = psr.write_kernel_times_json()
        meta_json = [{"class": x} for x in ["airplane", "automobile", "bird", "cat", "deer",
                                            "dog", "frog", "horse", "ship", "truck"]]
        with open("vars.js", "w") as wp:
            wp.write("var meta_json = %s;\n" % json.dumps(meta_json))
            wp.write("var samples_json = %s;\n" % json.dumps(samples_json))
            wp.write("var loss_accuracy_json = %s;\n" % json.dumps(loss_accuracy_json))
            wp.write("var kernel_times_json = %s;\n" % json.dumps(kernel_times_json))

def mainx():
    s = "array4<maxB, OC, H, W>& Convolution2D<maxB, IC, H, W, K, OC>::forward(array4<maxB, IC, H, W>&) [with int maxB = 64; int IC = 3; int H = 32; int W = 32; int K = 1; int OC = 16]"
    kp = kernel_parser()
    return kp.parse(s)

main()
