import pandas
import plotly
from plotly.offline import iplot
import plotly.express as px
import plotly.graph_objects as go


atom_names = {
        1: 'H',
        6: 'C',
        7: 'N',
        8: 'O',
        9: 'F',
        16: 'S',
        17: 'Cl',
    }

atom_colors = {
        'H': 'white',
        'C': 'black',
        'N': 'blue',
        'O': 'red',
        'F': 'green',
        'S': 'yellow',
        'Cl': 'green',
    }

def dgl_to_df(g):

    df = pandas.DataFrame()

    for feat in g.ndata.keys():
        arr = g.ndata[feat].detach().numpy()
        if arr.ndim == 2:
            for dim in range(arr.shape[1]):
                df[feat + str(dim)] = arr[:, dim]
        else:
            df[feat] = arr

    return df


def draw_plotly(g):

    df = dgl_to_df(g)


    ### Node trace ###
    names  = [atom_names[z] for z in df['attr5']]
    colors = [atom_colors[n] for n in names]

    node_trace=go.Scatter3d(
                x=df['pos0'],
                y=df['pos1'],
                z=df['pos2'],
                mode='markers',
                name='atom',
                marker=dict(symbol='circle',
                                size=df['attr5']*5,
                                color=colors,
                                line=dict(color='rgb(50,50,50)', width=2)
                                ),
                hovertemplate =
                 '<b>%{text}</b><br>', #+
                 #'<i>(eta,phi,lay)=(%{y:.2f},%{x:.2f},%{z:.2f})</i><br>',
                text = names #['{}<br>E=<i>{:.4f} GeV'.format(cl,en) for cl,en in zip(cell_df['cell_class_label'],cell_df['cell_e'])]
                )
    

    ### Edge trace ###
    g.apply_edges(lambda edges: {'x_src': edges.src['pos'][:,0]})
    g.apply_edges(lambda edges: {'x_dst': edges.dst['pos'][:,0]})
    g.apply_edges(lambda edges: {'y_src': edges.src['pos'][:,1]})
    g.apply_edges(lambda edges: {'y_dst': edges.dst['pos'][:,1]})
    g.apply_edges(lambda edges: {'z_src': edges.src['pos'][:,2]})
    g.apply_edges(lambda edges: {'z_dst': edges.dst['pos'][:,2]})

    x1list = g.edata['x_src'].detach().numpy()
    x2list = g.edata['x_dst'].detach().numpy()
    y1list = g.edata['y_src'].detach().numpy()
    y2list = g.edata['y_dst'].detach().numpy()
    z1list = g.edata['z_src'].detach().numpy()
    z2list = g.edata['z_dst'].detach().numpy()

    Xe,Ye,Ze = [],[],[]

    for eidx in range(len(x1list)):
        Xe += [x1list[eidx],x2list[eidx],None]
        Ye += [y1list[eidx],y2list[eidx],None]
        Ze += [z1list[eidx],z2list[eidx],None]

    edge_trace = go.Scatter3d(x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=dict(color='rgb(125,125,125,0.6)', width=3),
                hoverinfo='none'
                )
    

    ### Layout ###
    axis=dict(showbackground=False,
            showline=True,
            zeroline=False,
            showgrid=True,
            showticklabels=True,
            range=[-5,5],
            #title=var
            )

    layout = go.Layout(
            #title="Event "+str(event_idx),
            width=800,
            height=800,
            showlegend=False,
            scene=dict(
                xaxis=axis,
                yaxis=axis,
                zaxis=axis,
                aspectratio=dict(x=1, y=1, z=1),
            ),
        margin=dict(
            t=100
        ),
        hovermode='closest',
        #dragmode='pan',
        scene_xaxis_visible=True, scene_yaxis_visible=True, scene_zaxis_visible=True,
        legend=dict(font=dict(size=10),orientation='h'),
        )
    
    ### Plot ###
    data=[node_trace,edge_trace]
    fig=go.Figure(data=data, layout=layout)

    #fig.write_html("display.html")
    iplot(fig)