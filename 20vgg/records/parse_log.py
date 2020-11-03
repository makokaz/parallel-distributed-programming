#!/usr/bin/python3
import re
import sys

class log_parser_base:
    def __init__(self, fp):
        self.fp = fp
        self.tok_train_begin = "train_begin"
        self.tok_validate_begin = "validate_begin"
        self.tok_sample = "sample"
        self.tok_eof = "EOF"
        self.pat_train_begin = re.compile("(?P<t>\d+): === train (?P<a>\d+) - (?P<b>\d+) ===")
        self.pat_validate_begin = re.compile("(?P<t>\d+): === validate (?P<a>\d+) - (?P<b>\d+) ===")
        self.pat_sample = re.compile("(?P<t>\d+): sample (?P<sample>\d+) image (?P<image>\d+) pred (?P<pred>\d+) truth (?P<truth>\d+)")
        self.line_no = 0
        self.next_line()

    def next_line(self):
        """
        get next line, set token kind
        """
        self.line = self.fp.readline()
        if self.line == "":
            self.tok = self.tok_eof
            self.data = None
            return
        self.line_no += 1
        for tok, pat in [(self.tok_train_begin, self.pat_train_begin),
                         (self.tok_validate_begin, self.pat_validate_begin),
                         (self.tok_sample, self.pat_sample)]:
            m = pat.match(self.line)
            if m:
                self.tok = tok
                self.data = m.groupdict()
                return
        self.tok = None
        self.data = None
    def eat(self, tok):
        if self.tok != tok:
            self.parse_error(tok)
        self.next_line()
    def parse_error(self, tok):
        sys.stderr.write("%s:%d:error: expected %s but got %s\n" %
                         (self.fp.name, self.line_no, tok, self.tok))
        sys.exit(1)
    def parse_train(self):
        """
        === train 448 - 512 ===
        ...
        """
        self.action_train_begin(self.data)
        self.eat(self.tok_train_begin)
        while self.tok == None:
            self.eat(None)
        while self.tok == self.tok_sample:
            self.action_sample(self.data)
            self.eat(self.tok_sample)
        while self.tok == None:
            self.eat(None)
    def parse_validate(self):
        """
        === validate 448 - 512 ===
        ...
        """
        self.action_validate_begin(self.data)
        self.eat(self.tok_validate_begin)
        while self.tok == None:
            self.eat(None)
        while self.tok == self.tok_sample:
            self.action_sample(self.data)
            self.eat(self.tok_sample)
        while self.tok == None:
            self.eat(None)
    def parse_file(self):
        """
        other* train_or_validate* EOF
        """
        while self.tok == None:
            self.eat(None)
        while self.tok in [self.tok_train_begin, self.tok_validate_begin]:
            if self.tok == self.tok_train_begin:
                self.parse_train()
            elif self.tok == self.tok_validate_begin:
                self.parse_validate()
            else:
                assert(self.tok in [self.tok_train_begin,
                                    self.tok_validate_begin]), self.tok
        self.eat(self.tok_eof)

class log_parser(log_parser_base):
    def __init__(self, fp):
        super().__init__(fp)
        self.hist = []
    def action_train_begin(self, data):
        self.hist.append(("t", []))
    def action_validate_begin(self, data):
        self.hist.append(("v", []))
    def action_sample(self, data):
        t_or_v, cur = self.hist[-1]
        cur.append(data)
            
            
def main():
    log = sys.argv[1]
    print("iter,t_v,image,pred,truth")
    with open(log) as fp:
        psr = log_parser(fp)
        psr.parse_file()
        for i, (t_or_v, samples) in enumerate(psr.hist):
            for s in samples:
                print("%d,%s,%s,%s,%s"
                      % (i, t_or_v, s["image"], s["pred"], s["truth"]))

main()
