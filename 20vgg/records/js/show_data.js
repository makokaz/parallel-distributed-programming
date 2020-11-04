
/*  
 * ------------ ui components ------------ 
 */


/*  
 * ------------ data ------------ 
 */

var g_data;

function update_table_and_graph() {
    var div_id = "#history";
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
        td.append('img').attr("src", src).attr("title", title);
    }
}

function real_main(samples_csv, meta_csv) {
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
        };
        d3.selectAll("input").on("change", update_table_and_graph);
        update_table_and_graph();
    }
}

function main() {
    d3.csv('data/samples.csv').then(function (samples_csv) {
        d3.csv('data/batches.meta.txt').then(function (meta_csv) {
	    real_main(samples_csv, meta_csv);
        })
    });
}

main()
