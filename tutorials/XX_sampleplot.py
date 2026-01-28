        # sample along transect

        # indicative plot (2 panels; geospatial and along transect)
        if make_plot == 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9))

            # geospatial plot
            import matplotlib.colors as mcolors
            lats = [p["coordinates"][0] for p in transect_points]
            lons = [p["coordinates"][1] for p in transect_points]
            col_dict = {1: "Sand",2: "Mud",3: "Water",4: "Vegetation",5: "Other",6: "Turbid water",7: "Dry vegetation"}
            colors = ["#ffe30f", "#6d2c0c", "#3581eb", "#35b614", "#ff0101", "#0000ff", "#4f6701"]
            cmap = mcolors.ListedColormap(colors)
            labels = [p for p in class_sample_info["class"]]
            sc = ax1.scatter(lons, lats, c=np.array(labels) -1 , cmap=cmap)
            # Colorbar
            cbar = plt.colorbar(sc, ax=ax1)
            cbar.remove()
            #cbar.set_ticks(np.arange(len(col_dict)))        # 0–6
            #cbar.set_ticklabels(list(col_dict.values()))    # Sand, Mud, Water...

            # along transect plot
            newdist = list(np.arange(0, len(transect_points) * dist_steps, dist_steps))
            ax2.scatter(
                newdist,
                class_sample_info[class_sample_info == 1] * 0,
                color="#ffe30f",
                label="Sand",
                zorder=2,
            )
            ax2.scatter(
                newdist,
                class_sample_info[class_sample_info == 2] * 0,
                color="#6d2c0c",
                label="Mud",
                zorder=2,
            )
            ax2.scatter(
                newdist,
                class_sample_info[class_sample_info == 3] * 0,
                color="#3581eb",
                label="Water",
                zorder=2,
            )
            ax2.scatter(
                newdist,
                class_sample_info[class_sample_info == 6] * 0,
                color="#0000ff",
                label="Turbid water",
                zorder=2,
            )
            ax2.scatter(
                newdist,
                class_sample_info[class_sample_info == 4] * 0,
                color="#35b614",
                label="Vegetation",
                zorder=2,
            )
            ax2.scatter(
                newdist,
                class_sample_info[class_sample_info == 7] * 0,
                color="#4f6701",
                label="Dry vegetation",
                zorder=2,
            )
            ax2.scatter(
                newdist,
                class_sample_info[class_sample_info == 5] * 0,
                color="#ff0101",
                label="Other",
                zorder=2,
            )
            ax2.set_xlabel("Distance [m]", fontsize=16)
            #ax2.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            title = 'Transect profile ' + transect_id

            # save fig
            plt.savefig(os.path.join(figures, scriptname, transect_id), dpi = 300, bbox_inches='tight')
            plt.show()
