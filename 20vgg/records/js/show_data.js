
/*  
 * ------------ ui components ------------ 
 */


/*  
 * ------------ data ------------ 
 */

var g_data;

function update_loss_graph() {
    var W = { x : 1000, y : 500 };
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

function update_tables_and_graphs() {
    update_loss_graph();
    //update_samples_table();
}

function real_main(samples_csv, meta_csv, loss_accuracy_csv) {
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
        };
        d3.selectAll("input").on("change", update_tables_and_graphs);
        update_tables_and_graphs();
    }
}

function main() {
    d3.csv('data/samples.csv').then(function (samples_csv) {
        d3.csv('data/batches.meta.txt').then(function (meta_csv) {
            d3.csv('data/loss_accuracy.csv').then(function (loss_accuracy_csv) {
	        real_main(samples_csv, meta_csv, loss_accuracy_csv);
            })
        })
    });
}

main()
