
/*  
 * ------------ ui components ------------ 
 */


/*  
 * ------------ data ------------ 
 */

var g_data;

function update_loss_graph() {
    // graph area 1000x500
    var W = { x : 1000, y : 500 };
    // location of the originl (50,50)
    var O = { x : 50, y : 50 };
    var font_height = 10;

    var div_id = "#loss";
    d3.select(div_id).selectAll("*").remove();
    var data = g_data.loss_accuracy;
    // svg
    var svg = d3.select(div_id).append('svg')
        .attr('width',  W.x)
        .attr('height', W.y);
    // x, y functions
    var x = d3.scaleLinear();
    x.domain([0, d3.max(data, function(d) { return +d.samples; })])
        .range([ O.x, W.x ]);
    var y = d3.scaleLinear()
        .domain([0, d3.max(data, function(d) { return +d.train_loss; })])
        .range([ W.y - O.y, 0 ]);
    // x-axis
    var xaxis = svg.append('g') // create a <g> element
        .attr('class', 'x axis') // specify classes
        .attr('transform', 'translate(' + 0 + ',' + (W.y - O.y) + ')')
        .call(d3.axisBottom(x)); // let the axis do its thing
    // y-axis
    var yaxis = svg.append('g') // create a <g> element
        .attr('class', 'y axis') // specify classes
        .attr('transform', 'translate(' + O.x + ',' + 0 + ')')
        .call(d3.axisLeft(y)); // let the axis do its thing

    // line
    var line = d3.line()
        .x(function (d) { return x(d.samples); })
        .y(function (d) { return y(d.train_loss); });
    svg.append("path")
        .datum(data)
        .attr("fill", "none")
        .attr("stroke", "steelblue")
        .attr("stroke-width", 1.5)
        .attr("d", line);
    // x label
    svg.append('g') // create a <g> element
        .attr('class', 'x axis') // specify classes
        .attr('transform', 'translate(' + W.x/2 + ',' + (W.y - font_height) + ')')
        .append("text")
        .text("samples");
}

function update_samples_table() {
    var div_id = "#samples";
    d3.select(div_id).selectAll("*").remove();
    var table = d3.select(div_id).append('table').attr('border', 1);
    var tr = table.append('tr');
    var samples = g_data.samples;
    var meta = g_data.meta;
    /*
      var header = samples[0];
      var cols = {};              // column name -> col index
      for (var j = 0; j < header.length; j++) {
      var col = header[j];
      cols[col] = j;
      }
    */
    tr.append('th').text("iter");
    tr.append('th').text("train/validate");
    tr.append('th').text("correct");
    tr.append('th').text("wrong");
    var cur_iter = -1;
    var cur_tr = null;
    var cur_correct = null;
    var cur_wrong = null;
    for (var i = 0; i < samples.length; i++) {
        var s = samples[i];
        var pred = +s.pred;
        var truth = +s.truth;
        if (s.iter != cur_iter) {
            cur_iter = s.iter;
            cur_tr = table.append('tr');
            cur_tr.append("td").text(s.iter);
            cur_tr.append("td").text(s.t_v);
            cur_correct = cur_tr.append("td");
            cur_wrong = cur_tr.append("td");
        }
        var src = "imgs/i" + s.image.padStart(4, "0") + ".png";
        var title;
        var td;
        if (pred == truth) {
            td = cur_correct;
            title = s.image + "(" + meta[pred].class + ")";
        } else {
            td = cur_wrong;
            title = s.image + "(" + meta[pred].class + "," + meta[truth].class + ")";
        }
        td.append('img').attr("src", src).attr("width", 16).attr("title", title);
    }
}

function make_key_to_group(row, group_keys) {
    var key = "";
    for (var i = 0; i < group_keys.length; i++) {
        var col = group_keys[i];
        if (key != "") key = key.concat("::");
        key = key.concat(row[col]);
    }
    return key;
}

function summarize_kernel_times(kernel_times, group_keys, sort_keys, sort_dir) {
    var D = {};
    for (var i = 0; i < kernel_times.length; i++) {
        var row = kernel_times[i];
        var t0 = +row.t0;
        var t1 = +row.t1;
        var key = make_key_to_group(row, group_keys);
        if (!(key in D)) D[key] = [];
        D[key].push(t1 - t0)
    }
    var O = [];
    var keys = Object.keys(D);
    function make_keys(item) {
        var keys = [];
        for (var i = 0; i < sort_keys.length; i++) {
            var sk = sort_keys[i];
            var key = item[sk];
            keys.push(key)
        }
        return keys;
    }
    function sum(a) {
        return a.reduce(function(x, y){ return x + y; }, 0);
    }
    for (var i = 0; i < keys.length; i++) {
        var key = keys[i];
        var cls_fun = key.split("::");
        var cls   = cls_fun[0];
        var cargs = cls_fun[1];
        var fun   = cls_fun[2];
        var fargs = cls_fun[3];
        var ts    = D[key];
        var calls = ts.length;
        var total = sum(ts);
        var avg   = total / calls;
        var sigma = Math.sqrt(sum(ts.map(x => (x - avg) * (x - avg))));
        var item = {"cls" : cls, "cargs" : cargs,
                    "fun" : fun, "fargs" : fargs,
                    "calls" : calls, "total" : total, "avg" : avg, "sigma" : sigma };
        item["keys"] = make_keys(item);
        O.push(item)
    }
    function key_cmp(a, b) {
        for (var i = 0; i < a.keys.length; i++) {
            var ak = a.keys[i];
            var bk = b.keys[i];
            if (typeof ak == "number") {
                if (ak < bk) {
                    return -1;
                } else if (ak > bk) {
                    return 1;
                }
            } else {
                var s = ak.localeCompare(bk);
                if (s < 0) return -1;
                else if (s > 0) return 1;
            }
        }
        return 0;
    }
    O = O.sort(key_cmp);
    if (sort_dir == -1) {
        O = O.reverse();
    }
    return O;
}

function make_header(th, col) {
    th.text(col);
    th.append("a").attr("href", "javascript:void(0)")
        .on("click",
            function() {
                g_data.sort_keys = [col];
                g_data.sort_dir = 1;
                refresh_page();
            }).text("↑");
    th.append("text").text("/");
    th.append("a").attr("href", "javascript:void(0)")
        .on("click",
            function() {
                g_data.sort_keys = [col]
                g_data.sort_dir = -1;
                refresh_page();
            }).text("↓");
    return th;
}

function make_kernel_times_table(div_id, times) {
    d3.select(div_id).selectAll("*").remove();
    var table = d3.select(div_id).append('table').attr('border', 1);
    var tr = table.append('tr').selectAll()
        .data(["cls", "cargs", "fun", "fargs", "calls", "total", "avg", "sigma/avg"]).enter()
        .append("th").each(function (col) {
            var th = d3.select(this);
            make_header(th, col);
        });
    table.selectAll()
        .data(times)
        .enter()
        .append("tr")
        .each(function (row, i) {
            var tr = d3.select(this);
            var data = [row.cls, row.cargs, row.fun, row.fargs,
                        row.calls, row.total, row.avg.toFixed(2), (row.sigma/row.avg).toFixed(2)]
            tr.selectAll()
                .data(data)
                .enter()
                .append("td")
                .each(function (cell, i) {
                    var td = d3.select(this);
                    td.text(cell);
                });
        });
}

function update_kernel_times_table() {
    var kernel_times = g_data.kernel_times;
    var agg_times = summarize_kernel_times(kernel_times,
                                           g_data.group_keys, g_data.sort_keys, g_data.sort_dir);
    make_kernel_times_table("#kernel_times", agg_times);
}

function refresh_page() {
    update_loss_graph();
    update_samples_table();
    update_kernel_times_table();
}

function real_main(meta_csv, samples_csv, loss_accuracy_csv, kernel_times_csv) {
    if (samples_csv == null) {
        d3.select("#history")
            .append("p")
            .append("font")
            .attr("color", "red")
            .attr("size", "+5")
            .text("no data available (submit one)");
    } else {
        g_data = {
            samples : samples_csv,
            meta : meta_csv,
            loss_accuracy : loss_accuracy_csv,
            kernel_times : kernel_times_csv,
            group_keys : ["cls", "cargs", "fun", "fargs"],
            sort_keys : ["cls", "fun"],
            sort_dir : 1,
        };
        d3.selectAll("input").on("change", refresh_page);
        refresh_page();
    }
}

function main() {
    var meta_csv = [{"class": "airplane"},
                    {"class": "automobile"},
                    {"class": "bird"},
                    {"class": "cat"},
                    {"class": "deer"},
                    {"class": "dog"},
                    {"class": "frog"},
                    {"class": "horse"},
                    {"class": "ship"},
                    {"class": "truck"}];
    if (0) {
        d3.csv('data/samples.csv').then(function (samples_csv) {
            d3.csv('data/loss_accuracy.csv').then(function (loss_accuracy_csv) {
                d3.csv('data/kernel_times.csv').then(function (kernel_times_csv) {
	            real_main(meta_csv, samples_csv, loss_accuracy_csv, kernel_times_csv);
                })
            })
        });
    } else {
	real_main(meta_csv, samples_json, loss_accuracy_json, kernel_times_json);
    }
}

main()
