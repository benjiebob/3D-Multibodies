from exp_manager.vis_utils import get_visdom_connection, denorm_image_trivial
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import cv2
import os
from PIL import Image

class VisdomPlotly():
    def __init__(self, visdom_env_imgs, server, port, clear_env = True):
        self.visdom_env_imgs = visdom_env_imgs

        self.viz = get_visdom_connection(server=server,port=port)
        if not self.viz.check_connection():
            print("no visdom server! -> skipping batch vis")
            return
            
        if clear_env: # clear visualisations
            print("  ... clearing visdom environment")
            self.viz.close(env=visdom_env_imgs,win=None)

        self.camera = dict(
            up=dict(x=0, y=-1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0.0, y=0, z=2.0)
        )

        self.scene = dict(
            xaxis = dict(nticks=10, range=[-100,100],),
            yaxis = dict(nticks=10, range=[-100,100],),
            zaxis = dict(nticks=10, range=[-100,100],),
            camera = self.camera)

    def extend_to_skeleton(self, pt_cloud, skeleton, line_resolution = 25):
        ptcloud_now = pt_cloud
        for stick in skeleton:
            alpha = np.linspace(0,1,line_resolution)[:, None]
            linepoints = pt_cloud[stick[0],:][None,:] * alpha + \
                        pt_cloud[stick[1],:][None,:] * ( 1. - alpha )
            ptcloud_now = np.concatenate((ptcloud_now,linepoints),axis=0)

        return ptcloud_now

    def make_fig(self, rows, cols, epoch, it, idx_image, dataset, acc_detail):
        # fig_dict['subplot_title']
        title="e%d_it%d_im%d_%s"%(epoch, it, idx_image, dataset)

        self.fig = make_subplots(
            rows = rows, cols = cols, 
            specs=[[{"type": "xy"},{"type": "xy"},{"type": "scene"}, {"type": "xy"}]],
            subplot_titles=(
                title,
                "Projection",
                acc_detail,
                'Mode Freqs',
            ),
            column_widths=[0.5] * cols,
        )
 
    def add_image(self, img):
        bg_image = Image.fromarray(img)

        self.fig.update_layout(
            images = [
                go.layout.Image(
                    source=bg_image,
                    xref="x1",
                    yref="y1",
                    x=0,
                    y=bg_image.size[1],
                    sizex=bg_image.size[0],
                    sizey=bg_image.size[1],
                    sizing="stretch",
                    opacity=0.75,
                    layer="below"),

                go.layout.Image(
                    source=bg_image,
                    xref="x2",
                    yref="y2",
                    x=0,
                    y=bg_image.size[1],
                    sizex=bg_image.size[0],
                    sizey=bg_image.size[1],
                    sizing="stretch",
                    opacity=0.75,
                    layer="below")
            ]
        )

    def add_3d_points(self, pt_cloud_np, row, col, name, color, opacity=1.0, s=8, extend = False, visible=True, hide_text=False):
        if extend:
            pt_cloud_np = self.extend_to_skeleton(pt_cloud_np, SKELETON_3D)

        mode = 'markers+text'
        if hide_text:
            mode = 'markers'

        self.fig.add_trace(
            go.Scatter3d(
                x=-1 * pt_cloud_np[:, 0],
                y=pt_cloud_np[:, 1],
                z=pt_cloud_np[:, 2],
                mode=mode,
                text=[str(x) for x in range(pt_cloud_np.shape[0])],
                name=name,
                visible=visible,
                marker=dict(
                    size=s,
                    color=color,
                    opacity=opacity,
                )), row = row, col = col)

        self.fig.update_scenes(patch = self.scene, row = row, col = col)
        self.add_hack_points(row, col)


    # def add_mesh(self, verts, triangles, row, col, name, color):
    #     self.fig.add_trace(
    #         go.Mesh3d(
    #             x=verts[:, 0],
    #             y=verts[:, 1],
    #             z=verts[:, 2],
    #             colorbar_title='z',
    #             colorscale=[[0, 'gold'], 
    #                         [0.5, 'mediumturquoise'], 
    #                         [1, 'magenta']],
    #             # Intensity of each vertex, which will be interpolated and color-coded
    #             intensity=[0, 0.33, 0.66, 1],
    #             # i, j and k give the vertices of triangles
    #             i=triangles[:, 0],
    #             j=triangles[:, 1],
    #             k=triangles[:, 2],
    #             name=name,
    #             showscale=True
    #         )
    #     )
    #     self.fig.update_scenes(patch = self.scene, row = row, col = col)

    def add_2d_points(
        self, points, row, col, name, color, scale=6, 
        opacity=1.0, im_size = 224, extend=False, 
        visible=True, hide_text=False):
        points_npy = points

        if extend:
            points_npy = self.extend_to_skeleton(points_npy, SKELETON_2D)

        mode = 'markers+text'
        if hide_text:
            mode = 'markers'
                
        self.fig.add_trace(
            go.Scatter(
            x=points_npy[:, 0],
            y=im_size-points_npy[:, 1],
            mode=mode,
            text=[str(x) for x in range(points.shape[0])],
            name=name,
            visible=visible,
            marker=dict(
                size=scale,
                color=color,                # set color to an array/list of desired values
                opacity=opacity,
            )), row = row, col = col)

        self.fig.update_xaxes(range=[0, im_size], row=row, col=col)
        self.fig.update_yaxes(range=[0, im_size], row=row, col=col)

    def show(self):
        raw_size = 400
        self.fig.update_layout(height = raw_size, width = raw_size * 4)
        self.viz.plotlyplot(self.fig, env=self.visdom_env_imgs)

    def add_hack_points(self, row, col):
        hack_points = np.array([
            [-1000.0, -1000.0, -1000.0],
            [-1000.0, -1000.0, 1000.0],
            [-1000.0, 1000.0, -1000.0],
            [-1000.0, 1000.0, 1000.0],
            [1000.0, -1000.0, -1000.0],
            [1000.0, -1000.0, 1000.0],
            [1000.0, 1000.0, -1000.0],
            [1000.0, 1000.0, 1000.0]])

        self.fig.add_trace(
            go.Scatter3d(
                x=-1 * hack_points[:, 0],
                y=-1 * hack_points[:, 2],
                z=-1 * hack_points[:, 1],
                mode='markers',
                name='_fake_pts',
                visible=False,
                marker=dict(
                    size=1,
                    opacity = 0,
                    color=(0.0, 0.0, 0.0),
                )), row = row, col = col)

    def add_bar(self, stats, num_modes, row, col, name):
        freqs = np.bincount(stats, minlength=num_modes)
        fig = self.fig.add_trace(
            go.Bar(
                x=list(range(num_modes)), 
                y=freqs, 
                name=name), row = row, col = col)