def _switch_element_type(dimension):
    import StellBlanket as SB
    if dimension == 1:
        return SB.Lines.Line_Element
    elif dimension == 2:
        return SB.Triangles.Triangle_Element
    elif dimension == 3:
        return SB.Tetrahedrons.Tetrahedron_Element
    else:
        raise ValueError("Dimension must be 1, 2 or 3.")
    

def _test_domain_from_sb(dimension, degree):
    import StellBlanket as SB
    material_0 = SB.Generic_Elements.Material(SB.Generic_Elements.Constant_Cross_Section(0.673))
    material_1 = SB.Generic_Elements.Material(SB.Generic_Elements.Constant_Cross_Section(2.5))
    region_sizes=  [0.8, 0.2]
    resolutions  = [5,5]
    size_other_dim = 7.54
    resolutions_other_dim = 10
    element = _switch_element_type(dimension)
    cartesian_domain = SB.Generic_Elements.Create_Slab(element, region_sizes, resolutions, [material_0, material_1], size_other_dim, resolutions_other_dim)    
    return cartesian_domain

def _arbitrary_source_values(n_elements):
    return onp.linspace(1.0, 10.0, n_elements)


def _matfree_nte_from_sb(dimension, degree, tn_order, angle_index):
    import StellBlanket as SB
    domain   = _test_domain_from_sb(dimension, degree)

    energy_set = SB.Generic_Elements.Energy_Set([0,1])
    matfreente = SB.Generic_Elements.Create_RAM_Simulation(quad_set, energy_set, domain, degree, 0)

    source_values = _arbitrary_source_values(domain.connectivity.shape[0])
    sources = [SB.Generic_Elements.Constant_Source(_switch_element_type(dimension), val, energy_set[0]) for val in source_values]
    total_source = SB.Generic_Elements.Per_Element_Source(sources, domain)
    return matfreente, total_source

# domain = _test_domain_from_sb(3,2)
# import StellBlanket as SB
# quad_set = SB.Generic_Elements.TN_Quadrature_Set(4)
# _, source = _matfree_nte_from_sb(3, 2, 4, 0)
# simulation = SB.Generic_Elements.Create_RAM_Simulation(quad_set, SB.Generic_Elements.Energy_Set([0,1]), domain, 2, 0)
# simulation.simulation.Sweep(simulation.energy_set[0], simulation.quadrature_set[0], source)
# %timeit simulation.simulation.Total_Sweep(simulation.energy_set[0], source)