#!/usr/bin/python3
import re
import sys

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
        sys.stderr.write("%s:%d:error: expected %s but got %s\n" %
                         (self.fp.name, self.line_no, tok, self.tok))
        sys.stderr.write("[%s]\n" % self.line)
        sys.exit(1)
    def parse_mini_batch(self):
        while self.tok == None:
            self.eat(None)
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

class log_parser(log_parser_base):
    def __init__(self, fp):
        super().__init__(fp)
        self.samples = []
        self.n_training_samples = 0
        self.loss_acc = []
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
    def write_samples(self, filename):
        wp = open(filename, "w")
        wp.write("iter,t_v,image,pred,truth\n")
        for i, (t_or_v, samples) in enumerate(self.samples):
            for s in samples:
                wp.write("{i},{t_or_v},{image},{pred},{truth}\n"
                         .format(i=i, t_or_v=t_or_v, **s))
    def write_loss_accuracy(self, filename):
        wp = open(filename, "w")
        wp.write("samples,t,train_accuracy,validate_accuracy,train_loss,validate_loss\n")
        data = None
        for samples, kind, t, x in self.loss_acc:
            if data is None or data["samples"] != samples:
                if data is not None:
                    wp.write("{samples},{t},{train_accuracy},{validate_accuracy},{train_loss},{validate_loss}\n"
                             .format(**data))
                data = {"samples" : samples,
                        "t" : "",
                        "train_accuracy" : "",
                        "validate_accuracy" : "",
                        "train_loss" : "",
                        "validate_loss" : ""}
            data[kind] = x
            if kind == "train_accuracy":
                data["t"] = t
            
def main():
    log = sys.argv[1]
    with open(log) as fp:
        psr = log_parser(fp)
        psr.parse_file()
        psr.write_samples("samples.csv")
        psr.write_loss_accuracy("loss_accuracy.csv")

main()
