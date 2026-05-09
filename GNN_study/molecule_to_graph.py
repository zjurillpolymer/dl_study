"""Convert RDKit molecules to PyTorch Geometric graphs for GNN training"""

import torch
from rdkit import Chem
from torch_geometric.data import Data

# Atom feature helpers
ATOM_TYPES = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # B, C, N, O, F, P, S, Cl, Br, I
HYBRID_TYPES = [Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3]


def atom_features(atom):
    feats = []
    # (1) Atomic type one-hot (10 common + 1 for others)
    feats += [int(atom.GetAtomicNum() == t) for t in ATOM_TYPES]
    feats.append(int(atom.GetAtomicNum() not in ATOM_TYPES))

    # (2) Degree (0-5+)
    deg = min(atom.GetDegree(), 5)
    feats += [int(deg == i) for i in range(6)]

    # (3) Formal charge
    feats.append(atom.GetFormalCharge())

    # (4) Num implicit Hs (0-4)
    num_h = min(atom.GetTotalNumHs(), 4)
    feats += [int(num_h == i) for i in range(5)]

    # (5) Hybridization one-hot
    feats += [int(atom.GetHybridization() == h) for h in HYBRID_TYPES]

    # (6) Aromatic
    feats.append(int(atom.GetIsAromatic()))

    # (7) In ring
    feats.append(int(atom.IsInRing()))

    # (8) Chiral
    feats.append(int(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED))

    return feats


BOND_TYPES = [Chem.rdchem.BondType.SINGLE,
              Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE,
              Chem.rdchem.BondType.AROMATIC]


def bond_features(bond):
    """Extract edge features for an RDKit Bond."""
    feats = []
    # (1) Bond type one-hot
    feats += [int(bond.GetBondType() == bt) for bt in BOND_TYPES]
    # (2) Conjugated
    feats.append(int(bond.GetIsConjugated()))
    # (3) In ring
    feats.append(int(bond.IsInRing()))
    return feats


def mol_to_pyg_graph(mol):
    """Convert an RDKit Mol to a PyG Data graph with node & edge features."""
    if mol is None:
        return None

    # Node features
    node_feats = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()],
        dtype=torch.float32
    )

    # Edge list (undirected: both directions)
    src, dst = [], []
    edge_feats = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        feats = bond_features(bond)
        src += [u, v]
        dst += [v, u]
        edge_feats += [feats, feats]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_feats, dtype=torch.float32)

    return Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr)


def batch_from_smiles(smiles_list):
    """Convert a list of SMILES strings to a batched PyG graph."""
    from torch_geometric.data import Batch
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    graphs = [mol_to_pyg_graph(mol) for mol in mols if mol is not None]
    return Batch.from_data_list(graphs)


if __name__ == '__main__':
    # Quick test
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    g = mol_to_pyg_graph(mol)
    print(f"SMILES: {smiles}")
    print(f"Nodes: {g.num_nodes}, Edges: {g.num_edges}")
    print(f"Node feat shape: {g.x.shape}")
    print(f"Edge feat shape: {g.edge_attr.shape}")
    print(f"Edge index shape: {g.edge_index.shape}")
    print(f"Node features:\n{g.x}")
