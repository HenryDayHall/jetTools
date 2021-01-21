""" Module to add quick fake tracks and towers to an eventWise, based on pure mc truth """
import warnings
from ipdb import set_trace as st
import awkward
import numpy as np
import scipy.spatial
from . import PDGNames, Components


class Cylinder:
    def __init__(self, radius, endcap_pseudorapidity=2.5):
        self.radius = radius
        self._radius2 = radius**2
        self.endcap_pseudorapidity = endcap_pseudorapidity
        self._endcap_theta = 2*np.arctan(np.exp(-endcap_pseudorapidity))
        if self._endcap_theta == 0:  # the tube has no radius
            self._endcap_z = 0
        else:
            self._endcap_z = radius/np.tan(self._endcap_theta)

    def intersection_point(self, px, py, pz, x0=0., y0=0., z0=0.):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            px = np.array(px)
            py = np.array(py)
            pz = np.array(pz)
            x0 = np.array(x0)
            y0 = np.array(y0)
            z0 = np.array(z0)
            # first caclulate intersection z with infinite barrel
            inside = x0**2 + y0**2 < self._radius2
            a = (px**2 + py**2)
            reaches_barrel = inside*(a>0)
            b = 2*(px*x0 + py*y0)
            c = (x0**2 + y0**2 - self._radius2)
            m = np.zeros_like(pz, dtype=float)
            m[reaches_barrel] = ((-b + np.sqrt(b**2 - 4*a*c))/(2*a))[reaches_barrel]
            z = np.array(z0 + m*pz)
            # now find the theta to decide if this is on an endcap
            theta = np.arctan(self.radius/z)
            on_endcap = inside*(np.abs(theta)<self._endcap_theta)
            # if a is zero it will alos be on the endcap
            on_endcap += a==0
            # if it has no mothion in the z dir it cannot be on the endcap
            on_endcap *= np.abs(pz) > 0
            z[on_endcap] = np.sign(pz[on_endcap])*self._endcap_z
            m[on_endcap] = ((z-z0)/pz)[on_endcap]
            # now get the x and y
            x = x0 + m*px
            y = y0 + m*py
            inside = inside * (np.abs(z0) < self._endcap_z)
            assert np.allclose((x**2 + y**2)[inside*(~on_endcap)], self._radius2) 
            assert np.all((x**2 + y**2)[inside] <= self._radius2 + 1e-10) 
        return x, y, z, inside

    def add_cells(self, cell_width=None, cells_round_circumference=None):
        # start with the barrel cells
        circumference = 2*np.pi*self.radius
        if cell_width is None:
            cell_width = circumference/cells_round_circumference
        else:
            cells_round_circumference = int(np.ceil(circumference/cell_width))
        barrel_cell_phis = np.linspace(-np.pi+cell_width*0.5,
                                       np.pi-cell_width*0.5, cells_round_circumference)
        cells_along_barrel = int(np.ceil(2*self._endcap_z/cell_width))
        if cells_along_barrel == 1:
            barrel_cell_zs = np.zeros(1)
        else:
            barrel_cell_zs = np.linspace(-self._endcap_z+cell_width*0.5,
                                         self._endcap_z-cell_width*0.5, cells_along_barrel)
        barrel_cell_phis, barrel_cell_zs = np.array(np.meshgrid(barrel_cell_phis,
                                                           barrel_cell_zs)).reshape((2, -1))
        barrel_cell_xs = self.radius*np.cos(barrel_cell_phis)
        barrel_cell_ys = self.radius*np.sin(barrel_cell_phis)
        barrel_cells = np.vstack((barrel_cell_xs, barrel_cell_ys, barrel_cell_zs)).T
        # then get endcap cells
        endcap_xys = []
        cells_across_endcap = int(np.ceil(2*self.radius/cell_width))
        if cells_across_endcap == 1:
            endcap_xys = [[0, 0]]
        else:
            for x in np.linspace(-self.radius+cell_width*0.5,
                                 self.radius-cell_width*0.5, cells_across_endcap):
                for y in np.linspace(-self.radius+cell_width*0.5,
                                     self.radius-cell_width*0.5, cells_across_endcap):
                    if x**2 + y**2 <= self._radius2:
                        endcap_xys.append([x, y])
        n_endcaps = len(endcap_xys)
        endcap_xyzs_half1 = np.hstack((np.array(endcap_xys),
                                       np.full((n_endcaps, 1), self._endcap_z)))
        endcap_xyzs_half2 = np.hstack((np.array(endcap_xys),
                                       np.full((n_endcaps, 1), -self._endcap_z)))
        endcap_cells = np.vstack((endcap_xyzs_half1, endcap_xyzs_half2))
        self.cells = np.vstack((barrel_cells, endcap_cells))
        # now calculate the eta, phi coords
        self.cell_phis, cell_pts = Components.pxpy_to_phipt(self.cells[:, 0], self.cells[:, 1])
        self.cell_etas = Components.theta_to_pseudorapidity(
                            Components.ptpz_to_theta(cell_pts, self.cells[:, 2]))
        

    def closest_cell(self, px, py, pz, x0=0., y0=0., z0=0.):
        coords = np.array(self.intersection_point(px, py, pz, x0, y0, z0)[:3]).T
        coords = np.atleast_2d(coords)
        distances = scipy.spatial.distance.cdist(self.cells, coords)
        return np.argmin(distances, axis=0)


def append_tracks(eventWise, inner_radius=0.15, outer_radius=1.,
                  endcaps_rapidity=2.5, min_pt=0.5):
    """
    Tracks have,
    dX, dY, dZ (inner most point)
    X, Y, Z (vertex point)
    OuterX, OuterY, OuterZ (outer most point)
    Particle (gids of particles)
    """
    new_components = {name: [] for name in ["Particle",
                                            "dX", "dY", "dZ",
                                            "X", "Y", "Z",
                                            "OuterX", "OuterY", "OuterZ"]}
    inner_barrel = Cylinder(inner_radius, endcaps_rapidity)
    outer_barrel = Cylinder(outer_radius, endcaps_rapidity)
    mcpid_dict = PDGNames.Identities()
    eventWise.selected_index = None
    n_events = len(eventWise.X)
    for event_n in range(n_events):
        print(f"{event_n/n_events:.2%}", end='\r', flush=True)
        eventWise.selected_index = event_n
        # start by deciding which gids are relevent
        charged = np.fromiter((not np.isclose(mcpid_dict[pid]['charge'], 0)
                               for pid in eventWise.MCPID),
                              dtype=bool)
        makes_track = np.where(eventWise.Is_leaf * (eventWise.PT > min_pt) * charged)[0]
        new_components["Particle"].append(makes_track.tolist())
        # get the vertex
        vertex_barcodes = eventWise.Vertex_barcode.tolist()
        vertices = [vertex_barcodes.index(b) for b in
                    eventWise.Start_vertex_barcode[makes_track]]
        X = eventWise.X[vertices]
        Y = eventWise.Y[vertices]
        Z = eventWise.Z[vertices]
        new_components["X"].append(X)
        new_components["Y"].append(Y)
        new_components["Z"].append(Z)
        # using the vertices add in the inner and outer points
        px = eventWise.Px[makes_track]
        py = eventWise.Py[makes_track]
        pz = eventWise.Pz[makes_track]
        args = (px, py, pz, X, Y, Z)
        dX, dY, dZ, inside = inner_barrel.intersection_point(*args)
        new_components["dX"].append(dX)
        new_components["dY"].append(dY)
        new_components["dZ"].append(dZ)
        outerX, outerY, outerZ, _ = outer_barrel.intersection_point(*args)
        new_components["OuterX"].append(outerX)
        new_components["OuterY"].append(outerY)
        new_components["OuterZ"].append(outerZ)
        # filter everything to remove stuff that isn't inside
        for name in new_components:
            new_components[name][-1] = awkward.fromiter(new_components[name][-1])[inside]
    new_components = {"Track_" + name: awkward.fromiter(value) 
                      for name, value in new_components.items()}
    eventWise.append(**new_components)


def append_towers(eventWise, cells_round_circumference=200, radius=1.,
                  endcaps_rapidity=2.5, min_pt=0.5):
    """
    Towers have
    Eta, Phi, Energy
    Particles (gid of particles)
    """
    new_components = {name: [] for name in ["Particles",
                                            "Eta", "Phi", "Energy"]}
    barrel = Cylinder(radius, endcaps_rapidity)
    barrel.add_cells(cells_round_circumference=cells_round_circumference)
    n_cells = len(barrel.cells)
    eventWise.selected_index = None
    n_events = len(eventWise.X)
    for event_n in range(n_events):
        print(f"{event_n/n_events:.2%}", end='\r', flush=True)
        eventWise.selected_index = event_n
        # start by deciding which gids are relevent
        makes_tower = np.where(eventWise.Is_leaf * (eventWise.PT > min_pt))[0]
        # get the vertex
        vertex_barcodes = eventWise.Vertex_barcode.tolist()
        vertices = [vertex_barcodes.index(b) for b in
                    eventWise.Start_vertex_barcode[makes_tower]]
        X = eventWise.X[vertices]
        Y = eventWise.Y[vertices]
        Z = eventWise.Z[vertices]
        # using the vertices get the cell activated on the barrel
        px = eventWise.Px[makes_tower]
        py = eventWise.Py[makes_tower]
        pz = eventWise.Pz[makes_tower]
        cell_idxs = barrel.closest_cell(px, py, pz, X, Y, Z)
        cell_mask = [i in cell_idxs for i in range(n_cells)]
        tower_particles = awkward.fromiter([makes_tower[np.where(cell_idxs == i)[0]]
                                            for i in np.where(cell_mask)[0]])
        tower_energies = np.fromiter((np.sum(eventWise.Energy[ps])
                                      for ps in tower_particles),
                                     dtype=float)
        new_components["Particles"].append(tower_particles)
        new_components["Eta"].append(barrel.cell_etas[cell_mask])
        new_components["Phi"].append(barrel.cell_phis[cell_mask])
        new_components["Energy"].append(tower_energies)
    new_components = {"Tower_" + name: awkward.fromiter(value) 
                      for name, value in new_components.items()}
    eventWise.append(**new_components)


if __name__ == "__main__":
    from jet_tools import Components, InputTools
    eventWise_path = InputTools.get_file_name("Name the eventWise: ", '.awkd').strip()
    if eventWise_path:
        eventWise = Components.EventWise.from_file(eventWise_path)
        #print("Adding tracks")
        #append_tracks(eventWise)
        print("Adding towers")
        append_towers(eventWise)
    

