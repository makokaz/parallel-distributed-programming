
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
    var header = samples[0];
    var cols = {};              // column name -> col index
    for (var j = 0; j < header.length; j++) {
        var col = header[j];
        cols[col] = j;
    }
    tr.append('th').text("iter");
    tr.append('th').text("train/validate");
    tr.append('th').text("image");
    var cur_iter = -1;
    var cur_tr = null;
    var cur_td = null;
    for (var i = 1; i < samples.length; i++) {
        var sample = samples[i];
        var iter = sample[cols.iter];
        var t_v = sample[cols.t_v];
        var pred = sample[cols.pred];
        var truth = sample[cols.truth];
        if (iter != cur_iter) {
            cur_iter = iter;
            cur_tr = table.append('tr');
            cur_tr.append("td").text(iter);
            cur_tr.append("td").text(t_v);
            cur_td = cur_tr.append("td");
        }
        var image = 0; // sample[col.image];
        var src = "i" + image + ".png";
        var alt = image + "(" + pred + ")";
        cur_td.append('img').attr("src", src).attr("title", alt).attr("width", 40);
    }
}

function real_main(error, samples_csv) {
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
        };
        d3.selectAll("input").on("change", update_table_and_graph);
        update_table_and_graph();
    }
}

function main() {
    if (true) {
        var error = null;
        var samples_csv = d3.csvParseRows("iter,t_v,image,pred,truth\n"
                                          + "0,t,4921,9,3\n"
                                          + "0,t,1971,1,2\n"
                                          + "0,t,7160,1,5\n");
	real_main(error, samples_csv);
    } else {
        d3.csv('data/samples.csv',
               function (error, samples_csv) {
	           real_main(error, sample_csv);
               });
    }
}

main()
