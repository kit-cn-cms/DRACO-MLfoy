import rootpy.plotting as rp

# dictionary for colors
def get_plot_color( cls ):
    color_dict = {
        "ttH":   "royalblue",
        "ttlf":  "salmon",
        "ttcc":  "tomato",
        "ttbb":  "brown",
        "tt2b":  "darkred",
        "ttb":   "red",
        "False": "orange",
        "True":  "cyan"
        }
    if "ttH" in cls: cls = "ttH"
    return color_dict[cls]

# intialize plotting style
def init_plot_style():
    style = rp.style.get_style("ATLAS")
    style.SetEndErrorSize(3)
    rp.style.set_style(style)

# define style of signal histogram
def set_sig_hist_style( hist, cls ):
    hist.Sumw2()
    hist.markersize = 0
    hist.drawstyle = "shape"
    hist.legendstyle = "L"
    hist.fillstyle = "hollow"
    hist.linestyle = "solid"
    hist.linecolor = get_plot_color(cls)
    hist.linewidth = 2

# define style of background histogram
def set_bkg_hist_style( hist, cls ):
    hist.Sumw2()
    hist.markersize = 0
    hist.legendstyle = "F"
    hist.fillstyle = "solid"
    hist.fillcolor = get_plot_color(cls)
    hist.linecolor = "black"
    hist.linewidth = 1
    
# define style of data points
def set_data_hist_style( hist, cls ):
    print("TODO datastyle")
    return


# create canvas
def init_canvas( ratiopad = False):
    if not ratiopad:
        canvas = rp.Canvas(width = 1024, height = 768)
        canvas.SetTopMargin(0.07)
        canvas.SetBottomMargin(0.15)
        canvas.SetRightMargin(0.05)
        canvas.SetLeftMargin(0.15)
        canvas.SetTicks(1,1)
    else:
        print("TODO ratiopad")
        canvas = None

    
    return canvas

# create legend
def init_legend(hists):
    legend = rp.Legend(hists, entryheight = 0.03)
    legend.SetX1NDC(0.90)
    legend.SetX2NDC(0.99)
    legend.SetY1NDC(0.90)
    legend.SetY2NDC(0.99)
    legend.SetBorderSize(0)
    legend.SetLineStyle(0)
    legend.SetTextSize(0.04)
    legend.SetTextFont(42)
    legend.SetFillStyle(0)
    legend.Draw()
    
    return legend

def save_canvas(canvas, save_path, clear_canvas = True):
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(save_path)
    if clear_canvas: canvas.Clear()
