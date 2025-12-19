#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <cmath>
#include <sstream>
#include <iostream>
#include <set>
#include <map>

namespace py = pybind11;

struct Atom {
    std::string atomName;
    int residueNumber;
    char chain;
    double x, y, z;
    double bfactor;  // B-factor (pLDDT for AlphaFold structures)
};

// Parse ATOM/HETATM line from PDB format
Atom parseAtomLine(const std::string& line) {
    Atom atom;
    
    if (line.length() < 54) {
        throw std::runtime_error("Invalid ATOM line: too short");
    }
    
    atom.atomName = line.substr(12, 4);
    atom.atomName.erase(0, atom.atomName.find_first_not_of(" "));
    atom.atomName.erase(atom.atomName.find_last_not_of(" ") + 1);
    
    atom.chain = line[21];
    atom.residueNumber = std::stoi(line.substr(22, 4));
    atom.x = std::stod(line.substr(30, 8));
    atom.y = std::stod(line.substr(38, 8));
    atom.z = std::stod(line.substr(46, 8));
    
    // Parse B-factor (columns 61-66)
    if (line.length() >= 66) {
        try {
            atom.bfactor = std::stod(line.substr(60, 6));
        } catch (...) {
            atom.bfactor = 0.0;
        }
    } else {
        atom.bfactor = 0.0;
    }
    
    return atom;
}

// Check if atom is hydrogen
inline bool isHydrogen(const std::string& atomName) {
    return !atomName.empty() && atomName[0] == 'H';
}

// Calculate distance between two atoms
inline double calculateDistance(const Atom& a1, const Atom& a2) {
    double dx = a1.x - a2.x;
    double dy = a1.y - a2.y;
    double dz = a1.z - a2.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// Parse PDB and extract atoms with optional chain filter
std::vector<Atom> parseAtoms(const std::string& pdbText, 
                             bool excludeHydrogen = true,
                             const std::string& chain = "") {
    std::vector<Atom> atoms;
    std::istringstream stream(pdbText);
    std::string line;
    
    while (std::getline(stream, line)) {
        if (line.substr(0, 4) == "ATOM" || line.substr(0, 6) == "HETATM") {
            try {
                Atom atom = parseAtomLine(line);
                
                // Filter by chain if specified
                if (!chain.empty() && chain[0] != atom.chain) {
                    continue;
                }
                
                if (!excludeHydrogen || !isHydrogen(atom.atomName)) {
                    atoms.push_back(atom);
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not parse line: " << line << std::endl;
            }
        }
    }
    
    return atoms;
}

// Calculate full distance matrix (returns NumPy array)
py::array_t<double> calculateDistanceMatrix(const std::string& pdbText, 
                                           bool excludeHydrogen = true,
                                           const std::string& chain = "") {
    std::vector<Atom> atoms = parseAtoms(pdbText, excludeHydrogen, chain);
    size_t n = atoms.size();
    
    if (n == 0) {
        throw std::runtime_error("No atoms found matching criteria");
    }
    
    // Allocate NumPy array
    py::array_t<double> result({n, n});
    auto buf = result.mutable_unchecked<2>();
    
    // Calculate all pairwise distances
    for (size_t i = 0; i < n; ++i) {
        buf(i, i) = 0.0;  // Distance to self
        for (size_t j = i + 1; j < n; ++j) {
            double dist = calculateDistance(atoms[i], atoms[j]);
            buf(i, j) = dist;
            buf(j, i) = dist;  // Symmetric matrix
        }
    }
    
    return result;
}

// Calculate inter-chain distance matrix (chain1 vs chain2)
py::array_t<double> calculateInterChainDistanceMatrix(const std::string& pdbText,
                                                      const std::string& chain1,
                                                      const std::string& chain2,
                                                      bool excludeHydrogen = true) {
    if (chain1.empty() || chain2.empty()) {
        throw std::runtime_error("Both chain1 and chain2 must be specified");
    }
    
    std::vector<Atom> atoms1 = parseAtoms(pdbText, excludeHydrogen, chain1);
    std::vector<Atom> atoms2 = parseAtoms(pdbText, excludeHydrogen, chain2);
    
    size_t n1 = atoms1.size();
    size_t n2 = atoms2.size();
    
    if (n1 == 0 || n2 == 0) {
        throw std::runtime_error("No atoms found in one or both chains");
    }
    
    // Allocate NumPy array (n1 x n2, not symmetric if chains differ)
    py::array_t<double> result({n1, n2});
    auto buf = result.mutable_unchecked<2>();
    
    // Calculate distances between chains
    for (size_t i = 0; i < n1; ++i) {
        for (size_t j = 0; j < n2; ++j) {
            buf(i, j) = calculateDistance(atoms1[i], atoms2[j]);
        }
    }
    
    return result;
}

// Calculate residue-level distance matrix (min distance between any atoms)
py::array_t<double> calculateResidueDistanceMatrix(const std::string& pdbText,
                                                   bool excludeHydrogen = true,
                                                   const std::string& chain = "") {
    std::vector<Atom> atoms = parseAtoms(pdbText, excludeHydrogen, chain);
    
    if (atoms.empty()) {
        throw std::runtime_error("No atoms found matching criteria");
    }
    
    // Build residue index mapping
    std::vector<std::pair<char, int>> residues;  // (chain, residue_number)
    std::vector<int> atomToResidue;
    
    for (size_t i = 0; i < atoms.size(); ++i) {
        auto residueId = std::make_pair(atoms[i].chain, atoms[i].residueNumber);
        
        // Find or create residue index
        auto it = std::find(residues.begin(), residues.end(), residueId);
        if (it == residues.end()) {
            residues.push_back(residueId);
            atomToResidue.push_back(residues.size() - 1);
        } else {
            atomToResidue.push_back(std::distance(residues.begin(), it));
        }
    }
    
    size_t nRes = residues.size();
    
    // Initialize with large values
    py::array_t<double> result({nRes, nRes});
    auto buf = result.mutable_unchecked<2>();
    
    for (size_t i = 0; i < nRes; ++i) {
        for (size_t j = 0; j < nRes; ++j) {
            buf(i, j) = (i == j) ? 0.0 : 1e9;  // Large value for min
        }
    }
    
    // Calculate minimum distance between residue pairs
    for (size_t i = 0; i < atoms.size(); ++i) {
        for (size_t j = i + 1; j < atoms.size(); ++j) {
            int resI = atomToResidue[i];
            int resJ = atomToResidue[j];
            
            if (resI != resJ) {
                double dist = calculateDistance(atoms[i], atoms[j]);
                if (dist < buf(resI, resJ)) {
                    buf(resI, resJ) = dist;
                    buf(resJ, resI) = dist;
                }
            }
        }
    }
    
    return result;
}

// Calculate inter-chain residue distance matrix
py::array_t<double> calculateInterChainResidueDistanceMatrix(const std::string& pdbText,
                                                             const std::string& chain1,
                                                             const std::string& chain2,
                                                             bool excludeHydrogen = true) {
    if (chain1.empty() || chain2.empty()) {
        throw std::runtime_error("Both chain1 and chain2 must be specified");
    }
    
    std::vector<Atom> atoms1 = parseAtoms(pdbText, excludeHydrogen, chain1);
    std::vector<Atom> atoms2 = parseAtoms(pdbText, excludeHydrogen, chain2);
    
    if (atoms1.empty() || atoms2.empty()) {
        throw std::runtime_error("No atoms found in one or both chains");
    }
    
    // Build residue lists for each chain
    std::vector<int> residues1;
    std::vector<int> atomToResidue1;
    for (const auto& atom : atoms1) {
        auto it = std::find(residues1.begin(), residues1.end(), atom.residueNumber);
        if (it == residues1.end()) {
            residues1.push_back(atom.residueNumber);
            atomToResidue1.push_back(residues1.size() - 1);
        } else {
            atomToResidue1.push_back(std::distance(residues1.begin(), it));
        }
    }
    
    std::vector<int> residues2;
    std::vector<int> atomToResidue2;
    for (const auto& atom : atoms2) {
        auto it = std::find(residues2.begin(), residues2.end(), atom.residueNumber);
        if (it == residues2.end()) {
            residues2.push_back(atom.residueNumber);
            atomToResidue2.push_back(residues2.size() - 1);
        } else {
            atomToResidue2.push_back(std::distance(residues2.begin(), it));
        }
    }
    
    size_t nRes1 = residues1.size();
    size_t nRes2 = residues2.size();
    
    // Initialize with large values
    py::array_t<double> result({nRes1, nRes2});
    auto buf = result.mutable_unchecked<2>();
    
    for (size_t i = 0; i < nRes1; ++i) {
        for (size_t j = 0; j < nRes2; ++j) {
            buf(i, j) = 1e9;  // Large value for min
        }
    }
    
    // Calculate minimum distance between residue pairs across chains
    for (size_t i = 0; i < atoms1.size(); ++i) {
        for (size_t j = 0; j < atoms2.size(); ++j) {
            int resI = atomToResidue1[i];
            int resJ = atomToResidue2[j];
            
            double dist = calculateDistance(atoms1[i], atoms2[j]);
            if (dist < buf(resI, resJ)) {
                buf(resI, resJ) = dist;
            }
        }
    }
    
    return result;
}

// Get atom information as Python lists (for reference)
py::dict getAtomInfo(const std::string& pdbText, 
                    bool excludeHydrogen = true,
                    const std::string& chain = "") {
    std::vector<Atom> atoms = parseAtoms(pdbText, excludeHydrogen, chain);
    
    std::vector<std::string> atomNames;
    std::vector<int> residueNumbers;
    std::vector<char> chains;
    std::vector<double> x, y, z, bfactors;
    
    for (const auto& atom : atoms) {
        atomNames.push_back(atom.atomName);
        residueNumbers.push_back(atom.residueNumber);
        chains.push_back(atom.chain);
        x.push_back(atom.x);
        y.push_back(atom.y);
        z.push_back(atom.z);
        bfactors.push_back(atom.bfactor);
    }
    
    py::dict info;
    info["atom_names"] = atomNames;
    info["residue_numbers"] = residueNumbers;
    info["chains"] = py::cast(chains);
    info["x"] = x;
    info["y"] = y;
    info["z"] = z;
    info["bfactors"] = bfactors;
    info["n_atoms"] = atoms.size();
    
    return info;
}

// Calculate number of contacts (with chain options)
int calculateContacts(const std::string& pdbText, 
                     double cutoff = 4.5,
                     bool excludeHydrogen = true,
                     const std::string& chain = "",
                     const std::string& chain1 = "",
                     const std::string& chain2 = "",
                     double minPlddt = 0.0,
                     int minSeqDist = 0) {
    
    std::vector<Atom> atoms;
    std::vector<Atom> atoms_set1, atoms_set2;
    bool interChain = false;
    
    // Determine mode: single chain, inter-chain, or all chains
    if (!chain1.empty() && !chain2.empty()) {
        // Inter-chain mode
        atoms_set1 = parseAtoms(pdbText, excludeHydrogen, chain1);
        atoms_set2 = parseAtoms(pdbText, excludeHydrogen, chain2);
        interChain = true;
    } else if (!chain.empty()) {
        // Single chain mode
        atoms = parseAtoms(pdbText, excludeHydrogen, chain);
    } else {
        // All chains mode
        atoms = parseAtoms(pdbText, excludeHydrogen, "");
    }
    
    // Build residue-level pLDDT map (max pLDDT of any atom in residue)
    std::map<std::pair<char, int>, double> residuePlddt;
    
    if (interChain) {
        for (const auto& atom : atoms_set1) {
            auto key = std::make_pair(atom.chain, atom.residueNumber);
            if (residuePlddt.find(key) == residuePlddt.end() || atom.bfactor > residuePlddt[key]) {
                residuePlddt[key] = atom.bfactor;
            }
        }
        for (const auto& atom : atoms_set2) {
            auto key = std::make_pair(atom.chain, atom.residueNumber);
            if (residuePlddt.find(key) == residuePlddt.end() || atom.bfactor > residuePlddt[key]) {
                residuePlddt[key] = atom.bfactor;
            }
        }
    } else {
        for (const auto& atom : atoms) {
            auto key = std::make_pair(atom.chain, atom.residueNumber);
            if (residuePlddt.find(key) == residuePlddt.end() || atom.bfactor > residuePlddt[key]) {
                residuePlddt[key] = atom.bfactor;
            }
        }
    }
    
    std::set<std::pair<int, int>> contactingPairs;
    
    if (interChain) {
        // Inter-chain contacts
        for (size_t i = 0; i < atoms_set1.size(); ++i) {
            for (size_t j = 0; j < atoms_set2.size(); ++j) {
                // Check pLDDT filter
                auto key1 = std::make_pair(atoms_set1[i].chain, atoms_set1[i].residueNumber);
                auto key2 = std::make_pair(atoms_set2[j].chain, atoms_set2[j].residueNumber);
                
                if (residuePlddt[key1] < minPlddt || residuePlddt[key2] < minPlddt) {
                    continue;
                }
                
                double dist = calculateDistance(atoms_set1[i], atoms_set2[j]);
                
                if (dist < cutoff) {
                    int id1 = static_cast<int>(atoms_set1[i].chain) * 100000 + atoms_set1[i].residueNumber;
                    int id2 = static_cast<int>(atoms_set2[j].chain) * 100000 + atoms_set2[j].residueNumber;
                    
                    if (id1 > id2) std::swap(id1, id2);
                    contactingPairs.insert({id1, id2});
                }
            }
        }
    } else {
        // Intra-chain or all contacts
        for (size_t i = 0; i < atoms.size(); ++i) {
            for (size_t j = i + 1; j < atoms.size(); ++j) {
                // Skip atoms from the same residue
                if (atoms[i].residueNumber == atoms[j].residueNumber && 
                    atoms[i].chain == atoms[j].chain) {
                    continue;
                }
                
                // Check sequence distance (only for same chain)
                if (atoms[i].chain == atoms[j].chain) {
                    int seqDist = std::abs(atoms[i].residueNumber - atoms[j].residueNumber);
                    if (seqDist < minSeqDist) {
                        continue;
                    }
                }
                
                // Check pLDDT filter
                auto key1 = std::make_pair(atoms[i].chain, atoms[i].residueNumber);
                auto key2 = std::make_pair(atoms[j].chain, atoms[j].residueNumber);
                
                if (residuePlddt[key1] < minPlddt || residuePlddt[key2] < minPlddt) {
                    continue;
                }
                
                double dist = calculateDistance(atoms[i], atoms[j]);
                
                if (dist < cutoff) {
                    int id1 = static_cast<int>(atoms[i].chain) * 100000 + atoms[i].residueNumber;
                    int id2 = static_cast<int>(atoms[j].chain) * 100000 + atoms[j].residueNumber;
                    
                    if (id1 > id2) std::swap(id1, id2);
                    contactingPairs.insert({id1, id2});
                }
            }
        }
    }
    
    return contactingPairs.size();
}

// Calculate contact density
double calculateContactDensity(const std::string& pdbText,
                               double cutoff = 4.5,
                               bool excludeHydrogen = true,
                               const std::string& chain = "",
                               int minSeqDist = 5,
                               double minPlddt = 0.0) {
    std::vector<Atom> atoms = parseAtoms(pdbText, excludeHydrogen, chain);
    
    if (atoms.empty()) {
        throw std::runtime_error("No atoms found matching criteria");
    }
    
    // Build residue index mapping and pLDDT
    std::vector<std::pair<char, int>> residues;  // (chain, residue_number)
    std::vector<int> atomToResidue;
    std::map<std::pair<char, int>, double> residuePlddt;
    
    for (size_t i = 0; i < atoms.size(); ++i) {
        auto residueId = std::make_pair(atoms[i].chain, atoms[i].residueNumber);
        
        // Update pLDDT
        if (residuePlddt.find(residueId) == residuePlddt.end() || 
            atoms[i].bfactor > residuePlddt[residueId]) {
            residuePlddt[residueId] = atoms[i].bfactor;
        }
        
        // Find or create residue index
        auto it = std::find(residues.begin(), residues.end(), residueId);
        if (it == residues.end()) {
            residues.push_back(residueId);
            atomToResidue.push_back(residues.size() - 1);
        } else {
            atomToResidue.push_back(std::distance(residues.begin(), it));
        }
    }
    
    size_t nRes = residues.size();
    
    if (nRes == 0) {
        return 0.0;
    }
    
    // Initialize distance matrix with large values
    std::vector<std::vector<double>> distMatrix(nRes, std::vector<double>(nRes, 1e9));
    
    for (size_t i = 0; i < nRes; ++i) {
        distMatrix[i][i] = 0.0;
    }
    
    // Calculate minimum distance between residue pairs
    for (size_t i = 0; i < atoms.size(); ++i) {
        for (size_t j = i + 1; j < atoms.size(); ++j) {
            int resI = atomToResidue[i];
            int resJ = atomToResidue[j];
            
            if (resI != resJ) {
                double dist = calculateDistance(atoms[i], atoms[j]);
                if (dist < distMatrix[resI][resJ]) {
                    distMatrix[resI][resJ] = dist;
                    distMatrix[resJ][resI] = dist;
                }
            }
        }
    }
    
    // Count contacts
    int contactCount = 0;
    
    for (size_t i = 0; i < nRes; ++i) {
        for (size_t j = i + 1; j < nRes; ++j) {
            // Check pLDDT filter
            if (residuePlddt[residues[i]] < minPlddt || 
                residuePlddt[residues[j]] < minPlddt) {
                continue;
            }
            
            // Check sequence distance (only for same chain)
            if (residues[i].first == residues[j].first) {
                int seqDist = std::abs(residues[i].second - residues[j].second);
                if (seqDist < minSeqDist) {
                    continue;
                }
            }
            
            // Check distance cutoff
            if (distMatrix[i][j] < cutoff) {
                contactCount++;
            }
        }
    }
    
    // Calculate density: C / N
    double density = static_cast<double>(contactCount) / static_cast<double>(nRes);
    
    return density;
}

// Calculate interchain contacts
int calculateInterchainContacts(const std::string& pdbText,
                                const std::string& chain1,
                                const std::string& chain2,
                                double cutoff = 4.5,
                                bool excludeHydrogen = true,
                                double minPlddt = 0.0) {
    if (chain1.empty() || chain2.empty()) {
        throw std::runtime_error("Both chain1 and chain2 must be specified");
    }
    
    std::vector<Atom> atoms1 = parseAtoms(pdbText, excludeHydrogen, chain1);
    std::vector<Atom> atoms2 = parseAtoms(pdbText, excludeHydrogen, chain2);
    
    if (atoms1.empty() || atoms2.empty()) {
        throw std::runtime_error("No atoms found in one or both chains");
    }
    
    // Build residue-level pLDDT for both chains
    std::map<std::pair<char, int>, double> residuePlddt;
    
    for (const auto& atom : atoms1) {
        auto key = std::make_pair(atom.chain, atom.residueNumber);
        if (residuePlddt.find(key) == residuePlddt.end() || atom.bfactor > residuePlddt[key]) {
            residuePlddt[key] = atom.bfactor;
        }
    }
    for (const auto& atom : atoms2) {
        auto key = std::make_pair(atom.chain, atom.residueNumber);
        if (residuePlddt.find(key) == residuePlddt.end() || atom.bfactor > residuePlddt[key]) {
            residuePlddt[key] = atom.bfactor;
        }
    }
    
    std::set<std::pair<int, int>> contactingPairs;
    
    for (size_t i = 0; i < atoms1.size(); ++i) {
        for (size_t j = 0; j < atoms2.size(); ++j) {
            // Check pLDDT filter
            auto key1 = std::make_pair(atoms1[i].chain, atoms1[i].residueNumber);
            auto key2 = std::make_pair(atoms2[j].chain, atoms2[j].residueNumber);
            
            if (residuePlddt[key1] < minPlddt || residuePlddt[key2] < minPlddt) {
                continue;
            }
            
            double dist = calculateDistance(atoms1[i], atoms2[j]);
            
            if (dist < cutoff) {
                int id1 = static_cast<int>(atoms1[i].chain) * 100000 + atoms1[i].residueNumber;
                int id2 = static_cast<int>(atoms2[j].chain) * 100000 + atoms2[j].residueNumber;
                
                if (id1 > id2) std::swap(id1, id2);
                contactingPairs.insert({id1, id2});
            }
        }
    }
    
    return contactingPairs.size();
}

// Calculate interface pLDDT (average pLDDT of interface residues)
double calculateInterfacePlddt(const std::string& pdbText,
                               const std::string& chain1,
                               const std::string& chain2,
                               double cutoff = 4.5,
                               bool excludeHydrogen = true) {
    if (chain1.empty() || chain2.empty()) {
        throw std::runtime_error("Both chain1 and chain2 must be specified");
    }
    
    std::vector<Atom> atoms1 = parseAtoms(pdbText, excludeHydrogen, chain1);
    std::vector<Atom> atoms2 = parseAtoms(pdbText, excludeHydrogen, chain2);
    
    if (atoms1.empty() || atoms2.empty()) {
        throw std::runtime_error("No atoms found in one or both chains");
    }
    
    // Find interface residues (residues with any atom within cutoff)
    std::set<std::pair<char, int>> interfaceResidues;
    
    for (size_t i = 0; i < atoms1.size(); ++i) {
        for (size_t j = 0; j < atoms2.size(); ++j) {
            double dist = calculateDistance(atoms1[i], atoms2[j]);
            
            if (dist < cutoff) {
                interfaceResidues.insert(std::make_pair(atoms1[i].chain, atoms1[i].residueNumber));
                interfaceResidues.insert(std::make_pair(atoms2[j].chain, atoms2[j].residueNumber));
            }
        }
    }
    
    if (interfaceResidues.empty()) {
        return 0.0;
    }
    
    // Calculate average pLDDT of all atoms in interface residues
    double sumPlddt = 0.0;
    int atomCount = 0;
    
    for (const auto& atom : atoms1) {
        auto key = std::make_pair(atom.chain, atom.residueNumber);
        if (interfaceResidues.find(key) != interfaceResidues.end()) {
            sumPlddt += atom.bfactor;
            atomCount++;
        }
    }
    
    for (const auto& atom : atoms2) {
        auto key = std::make_pair(atom.chain, atom.residueNumber);
        if (interfaceResidues.find(key) != interfaceResidues.end()) {
            sumPlddt += atom.bfactor;
            atomCount++;
        }
    }
    
    if (atomCount == 0) {
        return 0.0;
    }
    
    return sumPlddt / atomCount;
}

// Pybind11 module definition
PYBIND11_MODULE(pdb_contacts, m) {
    m.doc() = "Fast PDB distance matrix and contact calculator";
    
    m.def("distance_matrix", &calculateDistanceMatrix,
          py::arg("pdb_text"),
          py::arg("exclude_hydrogen") = true,
          py::arg("chain") = "",
          "Calculate atom-level distance matrix from PDB text.\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  exclude_hydrogen: Skip hydrogen atoms (default: True)\n"
          "  chain: Calculate only for specific chain (default: all chains)\n\n"
          "Returns:\n"
          "  NxN NumPy array of distances in Angstroms");
    
    m.def("interchain_distance_matrix", &calculateInterChainDistanceMatrix,
          py::arg("pdb_text"),
          py::arg("chain1"),
          py::arg("chain2"),
          py::arg("exclude_hydrogen") = true,
          "Calculate atom-level distance matrix between two chains.\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  chain1: First chain identifier\n"
          "  chain2: Second chain identifier\n"
          "  exclude_hydrogen: Skip hydrogen atoms (default: True)\n\n"
          "Returns:\n"
          "  N1xN2 NumPy array of distances (chain1 atoms x chain2 atoms)");
    
    m.def("residue_distance_matrix", &calculateResidueDistanceMatrix,
          py::arg("pdb_text"),
          py::arg("exclude_hydrogen") = true,
          py::arg("chain") = "",
          "Calculate residue-level distance matrix (minimum distance between any atoms of two residues).\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  exclude_hydrogen: Skip hydrogen atoms (default: True)\n"
          "  chain: Calculate only for specific chain (default: all chains)\n\n"
          "Returns:\n"
          "  MxM NumPy array where M is number of residues");
    
    m.def("interchain_residue_distance_matrix", &calculateInterChainResidueDistanceMatrix,
          py::arg("pdb_text"),
          py::arg("chain1"),
          py::arg("chain2"),
          py::arg("exclude_hydrogen") = true,
          "Calculate residue-level distance matrix between two chains.\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  chain1: First chain identifier\n"
          "  chain2: Second chain identifier\n"
          "  exclude_hydrogen: Skip hydrogen atoms (default: True)\n\n"
          "Returns:\n"
          "  M1xM2 NumPy array (chain1 residues x chain2 residues)");
    
    m.def("get_atom_info", &getAtomInfo,
          py::arg("pdb_text"),
          py::arg("exclude_hydrogen") = true,
          py::arg("chain") = "",
          "Get atom information as Python dictionary.\n\n"
          "Parameters:\n"
          "  chain: Filter by specific chain (default: all chains)\n\n"
          "Returns:\n"
          "  Dictionary with keys: atom_names, residue_numbers, chains, x, y, z, n_atoms");
    
    m.def("calculate_contacts", &calculateContacts,
          py::arg("pdb_text"),
          py::arg("cutoff") = 4.5,
          py::arg("exclude_hydrogen") = true,
          py::arg("chain") = "",
          py::arg("chain1") = "",
          py::arg("chain2") = "",
          py::arg("min_plddt") = 0.0,
          py::arg("min_seq_dist") = 0,
          "Calculate number of residue contacts within cutoff distance.\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  cutoff: Distance cutoff in Angstroms (default: 4.5)\n"
          "  exclude_hydrogen: Skip hydrogen atoms (default: True)\n"
          "  chain: Calculate contacts within a single chain\n"
          "  chain1, chain2: Calculate contacts between two chains\n"
          "  min_plddt: Minimum pLDDT (B-factor) threshold for residues (default: 0.0)\n"
          "  min_seq_dist: Minimum sequence separation for contacts (default: 0)\n\n"
          "Returns:\n"
          "  Number of contacting residue pairs");
    
    m.def("contact_density", &calculateContactDensity,
          py::arg("pdb_text"),
          py::arg("cutoff") = 4.5,
          py::arg("exclude_hydrogen") = true,
          py::arg("chain") = "",
          py::arg("min_seq_dist") = 5,
          py::arg("min_plddt") = 0.0,
          "Calculate contact density (contacts per residue).\n\n"
          "Formula: CD = C / N, where C is number of contacts and N is number of residues.\n"
          "Contacts are defined as residue pairs with minimum atom distance < cutoff,\n"
          "sequence separation >= min_seq_dist, and pLDDT >= min_plddt.\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  cutoff: Distance cutoff in Angstroms (default: 4.5)\n"
          "  exclude_hydrogen: Skip hydrogen atoms (default: True)\n"
          "  chain: Calculate for specific chain (default: all chains)\n"
          "  min_seq_dist: Minimum sequence separation (default: 5)\n"
          "  min_plddt: Minimum pLDDT threshold (default: 0.0)\n\n"
          "Returns:\n"
          "  Contact density (float)");
    
    m.def("interchain_contacts", &calculateInterchainContacts,
          py::arg("pdb_text"),
          py::arg("chain1"),
          py::arg("chain2"),
          py::arg("cutoff") = 4.5,
          py::arg("exclude_hydrogen") = true,
          py::arg("min_plddt") = 0.0,
          "Calculate number of residue contacts between two chains.\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  chain1: First chain identifier\n"
          "  chain2: Second chain identifier\n"
          "  cutoff: Distance cutoff in Angstroms (default: 4.5)\n"
          "  exclude_hydrogen: Skip hydrogen atoms (default: True)\n"
          "  min_plddt: Minimum pLDDT threshold (default: 0.0)\n\n"
          "Returns:\n"
          "  Number of interchain residue contacts");
    
    m.def("interface_plddt", &calculateInterfacePlddt,
          py::arg("pdb_text"),
          py::arg("chain1"),
          py::arg("chain2"),
          py::arg("cutoff") = 4.5,
          py::arg("exclude_hydrogen") = true,
          "Calculate average pLDDT of interface residues.\n\n"
          "Interface residues are those with any atom within cutoff distance\n"
          "of the other chain. Returns average B-factor of all atoms in these residues.\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  chain1: First chain identifier\n"
          "  chain2: Second chain identifier\n"
          "  cutoff: Distance cutoff in Angstroms (default: 4.5)\n"
          "  exclude_hydrogen: Skip hydrogen atoms (default: True)\n\n"
          "Returns:\n"
          "  Average pLDDT (B-factor) of interface residues");
}