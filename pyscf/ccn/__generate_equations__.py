from cc_diagrams import coupled_cluster_networks, coupled_cluster_energy_networks, coupled_cluster_lambda_networks,\
    coupled_cluster_eom_networks, CoupledClusterNetwork
from codegen import CCDiagramCodeGenerator
from svg import CoupledClusterSVGLogger

with open("equations.py", 'w') as f:

    f.write("from util import e, p\n")

    # ------------
    # Ground state
    # ------------
    variants = (
        (1, "s", (4, 3)),
        (2, "sd", (8, 6)),
        ((2,), "d", (4, 3)),
        (3, "sdt", (12, 9)),
    )

    for order, tag, gs in variants:
        tag = "gs_" + tag
        print("Generating {}".format(tag))
        logger = CoupledClusterSVGLogger("log_{tag}.svg".format(tag=tag), gs)
        codegen = CCDiagramCodeGenerator()
        codegen.collect(
            map(CoupledClusterNetwork.resolved, coupled_cluster_networks(order)),
        )
        codegen.assign_input_names()
        f.write("\n\n{code}".format(code=codegen.assemble("eq_{tag}".format(tag=tag), logger=logger)))
        print("  total: {:d}".format(len(codegen.__diagram_destination__)))

        codegen = CCDiagramCodeGenerator()
        codegen.collect(
            map(CoupledClusterNetwork.resolved, coupled_cluster_energy_networks(order)),
        )
        codegen.assign_input_names()
        f.write("\n\n{code}".format(code=codegen.assemble("energy_{tag}".format(tag=tag), logger=logger)))
        print("  energy: {:d}".format(len(codegen.__diagram_destination__)))

    # ------------
    # Lambda
    # ------------

    for order, tag, gs in variants:
        tag = "lambda_"+tag
        print("Generating {}".format(tag))
        logger = CoupledClusterSVGLogger("log_{tag}.svg".format(tag=tag), gs)
        codegen = CCDiagramCodeGenerator()
        codegen.collect(map(CoupledClusterNetwork.resolved, coupled_cluster_lambda_networks(order)))
        codegen.assign_input_names()
        f.write("\n\n{code}".format(code=codegen.assemble("eq_{tag}".format(tag=tag), logger=logger)))
        print("  total: {:d}".format(len(codegen.__diagram_destination__)))

    # ------------
    # EOM
    # ------------

    for kind in ("ip", "ea"):
        for order, tag, gs in variants:
            tag = kind + "_" + tag
            print("Generating {}".format(tag))
            logger = CoupledClusterSVGLogger("log_{tag}.svg".format(tag=tag), gs)
            codegen = CCDiagramCodeGenerator()
            codegen.collect(map(CoupledClusterNetwork.resolved, coupled_cluster_eom_networks(order, kind)))
            codegen.assign_input_names()
            f.write("\n\n{code}".format(code=codegen.assemble("eq_{tag}".format(tag=tag), logger=logger)))
            print("  total: {:d}".format(len(codegen.__diagram_destination__)))
