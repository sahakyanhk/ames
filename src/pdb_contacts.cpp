#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <cmath>
#include <sstream>
#include <set>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <cctype>

namespace py = pybind11;

// ============================================================
// Data Structures
// ============================================================

struct ResidueId {
    char chain;
    int number;
    char insertionCode;

    bool operator==(const ResidueId& o) const {
        return chain == o.chain && number == o.number && insertionCode == o.insertionCode;
    }
    bool operator<(const ResidueId& o) const {
        if (chain != o.chain) return chain < o.chain;
        if (number != o.number) return number < o.number;
        return insertionCode < o.insertionCode;
    }
};

struct ResidueIdHash {
    size_t operator()(const ResidueId& r) const {
        size_t h = std::hash<char>()(r.chain);
        h ^= std::hash<int>()(r.number) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<char>()(r.insertionCode) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct Atom {
    std::string atomName;
    std::string residueName;  // 3-letter residue code (e.g., ALA, GLY)
    std::string element;      // Element symbol (e.g., "C", "N", "O", "S")
    int residueNumber;
    char chain;
    char insertionCode;       // PDB column 27 insertion code (' ' if none)
    double x, y, z;
    double bfactor;           // B-factor (pLDDT for AlphaFold structures)

    ResidueId residueId() const {
        return {chain, residueNumber, insertionCode};
    }
};

struct ResidueIndex {
    std::vector<ResidueId> residues;            // ordered, first-seen
    std::vector<int> atomToResidue;             // atom index -> residue index
    std::unordered_map<ResidueId, int, ResidueIdHash> residueToIndex;  // O(1) lookup
};

// ============================================================
// Helpers
// ============================================================

// Parse ATOM/HETATM line from PDB format
Atom parseAtomLine(const std::string& line) {
    Atom atom;

    if (line.length() < 54) {
        throw std::runtime_error("Invalid ATOM line: too short");
    }

    atom.atomName = line.substr(12, 4);
    atom.atomName.erase(0, atom.atomName.find_first_not_of(" "));
    atom.atomName.erase(atom.atomName.find_last_not_of(" ") + 1);

    // Parse residue name (columns 18-20, 3-letter code)
    atom.residueName = line.substr(17, 3);
    atom.residueName.erase(0, atom.residueName.find_first_not_of(" "));
    atom.residueName.erase(atom.residueName.find_last_not_of(" ") + 1);

    atom.chain = line[21];
    atom.residueNumber = std::stoi(line.substr(22, 4));
    atom.insertionCode = line[26];  // column 27 (1-indexed), ' ' if none

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

    // Parse element symbol (columns 77-78, or derive from atom name)
    if (line.length() >= 78) {
        std::string elem = line.substr(76, 2);
        elem.erase(0, elem.find_first_not_of(" "));
        elem.erase(elem.find_last_not_of(" ") + 1);
        if (!elem.empty()) {
            elem[0] = std::toupper(static_cast<unsigned char>(elem[0]));
            if (elem.size() > 1)
                elem[1] = std::tolower(static_cast<unsigned char>(elem[1]));
            atom.element = elem;
        }
    }
    if (atom.element.empty()) {
        for (size_t i = 0; i < atom.atomName.size(); ++i) {
            if (std::isalpha(static_cast<unsigned char>(atom.atomName[i]))) {
                atom.element = std::string(1, std::toupper(static_cast<unsigned char>(atom.atomName[i])));
                break;
            }
        }
        if (atom.element.empty()) atom.element = "C";
    }

    return atom;
}

// Check if atom is hydrogen
// Handles both modern (H, HA, HB2) and old-style (1H, 2HG, 3HD1) naming
inline bool isHydrogen(const std::string& atomName) {
    if (atomName.empty()) return false;
    if (atomName[0] == 'H') return true;
    // Old-style: digit prefix followed by H (e.g., 1H, 2HG)
    size_t pos = 0;
    while (pos < atomName.size() && std::isdigit(static_cast<unsigned char>(atomName[pos]))) {
        pos++;
    }
    return pos > 0 && pos < atomName.size() && atomName[pos] == 'H';
}

// Calculate distance between two atoms
inline double calculateDistance(const Atom& a1, const Atom& a2) {
    double dx = a1.x - a2.x;
    double dy = a1.y - a2.y;
    double dz = a1.z - a2.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// VDW radius lookup by element symbol (Bondi radii)
inline double getVdwRadius(const std::string& element) {
    static const std::unordered_map<std::string, double> radii = {
        {"C",  1.70}, {"N",  1.55}, {"O",  1.52}, {"S",  1.80},
        {"H",  1.20}, {"P",  1.80}, {"Se", 1.90}, {"F",  1.47},
        {"Cl", 1.75}, {"Br", 1.85}, {"I",  1.98}, {"Zn", 1.39},
        {"Fe", 1.63}, {"Mg", 1.73}, {"Ca", 1.74}, {"Na", 1.02},
        {"K",  1.38}
    };
    auto it = radii.find(element);
    if (it != radii.end()) return it->second;
    return 1.70; // Default: carbon radius
}

// Check if two elements can form a hydrogen bond (donor-acceptor)
inline bool canHbond(const std::string& elem1, const std::string& elem2) {
    auto isDonorAcceptor = [](const std::string& e) {
        return e == "N" || e == "O" || e == "S";
    };
    return isDonorAcceptor(elem1) && isDonorAcceptor(elem2);
}

// Check if atom's chain matches the comma-separated chain filter.
// e.g. "A" matches chain A, "A,B" matches chains A or B.
// "AB" is a single chain ID (not two chains).
// Empty filter matches all chains.
inline bool chainMatches(char atomChain, const std::string& chainFilter) {
    if (chainFilter.empty()) return true;
    size_t start = 0;
    while (start <= chainFilter.size()) {
        size_t end = chainFilter.find(',', start);
        if (end == std::string::npos) end = chainFilter.size();
        if (end - start == 1 && chainFilter[start] == atomChain) return true;
        start = end + 1;
    }
    return false;
}

// Parse PDB and extract atoms with optional chain filter
// chain: comma-separated chain IDs (e.g., "A,B" matches chains A and B)
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

                if (!chainMatches(atom.chain, chain)) {
                    continue;
                }

                if (!excludeHydrogen || !isHydrogen(atom.atomName)) {
                    atoms.push_back(atom);
                }
            } catch (...) {
                // Skip unparsable ATOM/HETATM lines silently
            }
        }
    }

    return atoms;
}

// Build residue index from atoms (O(1) lookup via unordered_map)
ResidueIndex buildResidueIndex(const std::vector<Atom>& atoms) {
    ResidueIndex idx;
    for (size_t i = 0; i < atoms.size(); ++i) {
        ResidueId rid = atoms[i].residueId();
        auto it = idx.residueToIndex.find(rid);
        if (it == idx.residueToIndex.end()) {
            int newIdx = static_cast<int>(idx.residues.size());
            idx.residues.push_back(rid);
            idx.residueToIndex[rid] = newIdx;
            idx.atomToResidue.push_back(newIdx);
        } else {
            idx.atomToResidue.push_back(it->second);
        }
    }
    return idx;
}

// ============================================================
// Distance Matrices
// ============================================================

// Calculate full distance matrix (returns NumPy array)
py::array_t<double> calculateDistanceMatrix(const std::string& pdbText,
                                           bool excludeHydrogen = true,
                                           const std::string& chain = "") {
    std::vector<Atom> atoms = parseAtoms(pdbText, excludeHydrogen, chain);
    size_t n = atoms.size();

    if (n == 0) {
        throw std::runtime_error("No atoms found matching criteria");
    }

    py::array_t<double> result({n, n});
    auto buf = result.mutable_unchecked<2>();

    for (size_t i = 0; i < n; ++i) {
        buf(i, i) = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            double dist = calculateDistance(atoms[i], atoms[j]);
            buf(i, j) = dist;
            buf(j, i) = dist;
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
        throw std::runtime_error("No atoms found in one or both chain groups");
    }

    py::array_t<double> result({n1, n2});
    auto buf = result.mutable_unchecked<2>();

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

    ResidueIndex ridx = buildResidueIndex(atoms);
    size_t nRes = ridx.residues.size();

    py::array_t<double> result({nRes, nRes});
    auto buf = result.mutable_unchecked<2>();

    for (size_t i = 0; i < nRes; ++i) {
        for (size_t j = 0; j < nRes; ++j) {
            buf(i, j) = (i == j) ? 0.0 : 1e9;
        }
    }

    for (size_t i = 0; i < atoms.size(); ++i) {
        for (size_t j = i + 1; j < atoms.size(); ++j) {
            int resI = ridx.atomToResidue[i];
            int resJ = ridx.atomToResidue[j];

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
        throw std::runtime_error("No atoms found in one or both chain groups");
    }

    ResidueIndex ridx1 = buildResidueIndex(atoms1);
    ResidueIndex ridx2 = buildResidueIndex(atoms2);

    size_t nRes1 = ridx1.residues.size();
    size_t nRes2 = ridx2.residues.size();

    py::array_t<double> result({nRes1, nRes2});
    auto buf = result.mutable_unchecked<2>();

    for (size_t i = 0; i < nRes1; ++i) {
        for (size_t j = 0; j < nRes2; ++j) {
            buf(i, j) = 1e9;
        }
    }

    for (size_t i = 0; i < atoms1.size(); ++i) {
        for (size_t j = 0; j < atoms2.size(); ++j) {
            int resI = ridx1.atomToResidue[i];
            int resJ = ridx2.atomToResidue[j];

            double dist = calculateDistance(atoms1[i], atoms2[j]);
            if (dist < buf(resI, resJ)) {
                buf(resI, resJ) = dist;
            }
        }
    }

    return result;
}

// ============================================================
// Atom / Residue Info
// ============================================================

// Get atom information as Python dictionary
py::dict getAtomInfo(const std::string& pdbText,
                    bool excludeHydrogen = true,
                    const std::string& chain = "") {
    std::vector<Atom> atoms = parseAtoms(pdbText, excludeHydrogen, chain);

    std::vector<std::string> atomNames;
    std::vector<std::string> residueNames;
    std::vector<int> residueNumbers;
    std::vector<char> chains;
    std::vector<double> x, y, z, bfactors;

    // Build unique residue list (insertion-ordered)
    std::vector<std::pair<ResidueId, std::string>> uniqueResidues;
    std::unordered_map<ResidueId, int, ResidueIdHash> seenResidues;

    for (const auto& atom : atoms) {
        atomNames.push_back(atom.atomName);
        residueNames.push_back(atom.residueName);
        residueNumbers.push_back(atom.residueNumber);
        chains.push_back(atom.chain);
        x.push_back(atom.x);
        y.push_back(atom.y);
        z.push_back(atom.z);
        bfactors.push_back(atom.bfactor);

        ResidueId rid = atom.residueId();
        if (seenResidues.find(rid) == seenResidues.end()) {
            seenResidues[rid] = static_cast<int>(uniqueResidues.size());
            uniqueResidues.push_back({rid, atom.residueName});
        }
    }

    // Build residue info lists
    std::vector<char> residueChains;
    std::vector<int> residueNums;
    std::vector<std::string> residueNamesList;

    for (const auto& [rid, resName] : uniqueResidues) {
        residueChains.push_back(rid.chain);
        residueNums.push_back(rid.number);
        residueNamesList.push_back(resName);
    }

    // Build sequence string per chain
    std::map<char, std::string> chainSequences;
    std::map<std::string, char> threeToOne = {
        {"ALA", 'A'}, {"ARG", 'R'}, {"ASN", 'N'}, {"ASP", 'D'}, {"CYS", 'C'},
        {"GLN", 'Q'}, {"GLU", 'E'}, {"GLY", 'G'}, {"HIS", 'H'}, {"ILE", 'I'},
        {"LEU", 'L'}, {"LYS", 'K'}, {"MET", 'M'}, {"PHE", 'F'}, {"PRO", 'P'},
        {"SER", 'S'}, {"THR", 'T'}, {"TRP", 'W'}, {"TYR", 'Y'}, {"VAL", 'V'},
        // Non-standard amino acids
        {"SEC", 'U'}, {"PYL", 'O'}, {"ASX", 'B'}, {"GLX", 'Z'}, {"XLE", 'J'},
        {"UNK", 'X'}
    };

    for (const auto& [rid, resName] : uniqueResidues) {
        char oneLetterCode = 'X';
        auto it = threeToOne.find(resName);
        if (it != threeToOne.end()) {
            oneLetterCode = it->second;
        }
        chainSequences[rid.chain] += oneLetterCode;
    }

    // Convert chain sequences to Python dict
    py::dict sequences;
    for (const auto& [ch, seq] : chainSequences) {
        sequences[py::cast(std::string(1, ch))] = seq;
    }

    py::dict info;
    // Atom-level info
    info["atom_names"] = atomNames;
    info["residue_names"] = residueNames;
    info["residue_numbers"] = residueNumbers;
    info["chains"] = py::cast(chains);
    info["x"] = x;
    info["y"] = y;
    info["z"] = z;
    info["bfactors"] = bfactors;
    info["n_atoms"] = atoms.size();

    // Residue-level info
    info["residues"] = py::dict(
        py::arg("chains") = py::cast(residueChains),
        py::arg("numbers") = residueNums,
        py::arg("names") = residueNamesList,
        py::arg("n_residues") = uniqueResidues.size()
    );
    info["sequences"] = sequences;

    return info;
}

// ============================================================
// Contact Counting
// ============================================================

// Calculate number of contacts (with chain options)
// pLDDT filtering is atom-level: both atoms forming a contact must have bfactor >= minPlddt
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
        atoms_set1 = parseAtoms(pdbText, excludeHydrogen, chain1);
        atoms_set2 = parseAtoms(pdbText, excludeHydrogen, chain2);
        interChain = true;
    } else if (!chain.empty()) {
        atoms = parseAtoms(pdbText, excludeHydrogen, chain);
    } else {
        atoms = parseAtoms(pdbText, excludeHydrogen, "");
    }

    // Use set of ResidueId pairs for collision-free counting
    std::set<std::pair<ResidueId, ResidueId>> contactingPairs;

    auto addContact = [&](const Atom& a1, const Atom& a2) {
        ResidueId rid1 = a1.residueId();
        ResidueId rid2 = a2.residueId();
        if (rid2 < rid1) std::swap(rid1, rid2);
        contactingPairs.insert({rid1, rid2});
    };

    if (interChain) {
        for (size_t i = 0; i < atoms_set1.size(); ++i) {
            for (size_t j = 0; j < atoms_set2.size(); ++j) {
                // Atom-level pLDDT filter
                if (atoms_set1[i].bfactor < minPlddt || atoms_set2[j].bfactor < minPlddt) {
                    continue;
                }

                double dist = calculateDistance(atoms_set1[i], atoms_set2[j]);

                if (dist < cutoff) {
                    addContact(atoms_set1[i], atoms_set2[j]);
                }
            }
        }
    } else {
        for (size_t i = 0; i < atoms.size(); ++i) {
            for (size_t j = i + 1; j < atoms.size(); ++j) {
                // Skip atoms from the same residue
                if (atoms[i].residueId() == atoms[j].residueId()) {
                    continue;
                }

                // Check sequence distance (only for same chain)
                if (atoms[i].chain == atoms[j].chain) {
                    int seqDist = std::abs(atoms[i].residueNumber - atoms[j].residueNumber);
                    if (seqDist < minSeqDist) {
                        continue;
                    }
                }

                // Atom-level pLDDT filter
                if (atoms[i].bfactor < minPlddt || atoms[j].bfactor < minPlddt) {
                    continue;
                }

                double dist = calculateDistance(atoms[i], atoms[j]);

                if (dist < cutoff) {
                    addContact(atoms[i], atoms[j]);
                }
            }
        }
    }

    return contactingPairs.size();
}

// ============================================================
// Contact Density
// ============================================================

// Calculate contact density (contacts per residue)
// pLDDT filtering is atom-level in the distance computation
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

    ResidueIndex ridx = buildResidueIndex(atoms);
    size_t nRes = ridx.residues.size();

    if (nRes == 0) {
        return 0.0;
    }

    // Build residue min-distance matrix
    // Only atom pairs where both atoms pass pLDDT contribute
    std::vector<std::vector<double>> distMatrix(nRes, std::vector<double>(nRes, 1e9));

    for (size_t i = 0; i < nRes; ++i) {
        distMatrix[i][i] = 0.0;
    }

    for (size_t i = 0; i < atoms.size(); ++i) {
        for (size_t j = i + 1; j < atoms.size(); ++j) {
            int resI = ridx.atomToResidue[i];
            int resJ = ridx.atomToResidue[j];

            if (resI != resJ) {
                // Atom-level pLDDT filter
                if (atoms[i].bfactor < minPlddt || atoms[j].bfactor < minPlddt) {
                    continue;
                }

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
            // Check sequence distance (only for same chain)
            if (ridx.residues[i].chain == ridx.residues[j].chain) {
                int seqDist = std::abs(ridx.residues[i].number - ridx.residues[j].number);
                if (seqDist < minSeqDist) {
                    continue;
                }
            }

            if (distMatrix[i][j] < cutoff) {
                contactCount++;
            }
        }
    }

    // Density: C / N
    return static_cast<double>(contactCount) / static_cast<double>(nRes);
}

// ============================================================
// Interchain Contacts (convenience wrapper)
// ============================================================

// Delegates to calculateContacts
int calculateInterchainContacts(const std::string& pdbText,
                                const std::string& chain1,
                                const std::string& chain2,
                                double cutoff = 4.5,
                                bool excludeHydrogen = true,
                                double minPlddt = 0.0) {
    if (chain1.empty() || chain2.empty()) {
        throw std::runtime_error("Both chain1 and chain2 must be specified");
    }
    return calculateContacts(pdbText, cutoff, excludeHydrogen, "", chain1, chain2, minPlddt, 0);
}

// ============================================================
// Ligand Contacts
// ============================================================

// Calculate ligand contact density (contacts / number of ligand atoms)
double calculateLigandContactDensity(const std::string& pdbText,
                                     const std::string& polymerChain,
                                     const std::string& ligandChain,
                                     double cutoff = 4.5,
                                     bool excludeHydrogen = true,
                                     double minPlddt = 0.0) {
    if (polymerChain.empty() || ligandChain.empty()) {
        throw std::runtime_error("Both polymer_chain and ligand_chain must be specified");
    }

    std::vector<Atom> polymerAtoms = parseAtoms(pdbText, excludeHydrogen, polymerChain);
    std::vector<Atom> ligandAtoms = parseAtoms(pdbText, excludeHydrogen, ligandChain);

    if (polymerAtoms.empty()) {
        throw std::runtime_error("No atoms found in polymer chain(s)");
    }
    if (ligandAtoms.empty()) {
        throw std::runtime_error("No atoms found in ligand chain(s)");
    }

    size_t nLigandAtoms = ligandAtoms.size();

    // Count residue-pair contacts
    std::set<std::pair<ResidueId, ResidueId>> contactingPairs;

    for (size_t i = 0; i < polymerAtoms.size(); ++i) {
        for (size_t j = 0; j < ligandAtoms.size(); ++j) {
            // Atom-level pLDDT filter
            if (polymerAtoms[i].bfactor < minPlddt || ligandAtoms[j].bfactor < minPlddt) {
                continue;
            }

            double dist = calculateDistance(polymerAtoms[i], ligandAtoms[j]);

            if (dist < cutoff) {
                ResidueId rid1 = polymerAtoms[i].residueId();
                ResidueId rid2 = ligandAtoms[j].residueId();
                if (rid2 < rid1) std::swap(rid1, rid2);
                contactingPairs.insert({rid1, rid2});
            }
        }
    }

    return static_cast<double>(contactingPairs.size()) / static_cast<double>(nLigandAtoms);
}

// ============================================================
// Interface pLDDT
// ============================================================

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
        throw std::runtime_error("No atoms found in one or both chain groups");
    }

    // Find interface residues (residues with any atom within cutoff)
    std::set<ResidueId> interfaceResidues;

    for (size_t i = 0; i < atoms1.size(); ++i) {
        for (size_t j = 0; j < atoms2.size(); ++j) {
            double dist = calculateDistance(atoms1[i], atoms2[j]);

            if (dist < cutoff) {
                interfaceResidues.insert(atoms1[i].residueId());
                interfaceResidues.insert(atoms2[j].residueId());
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
        if (interfaceResidues.count(atom.residueId())) {
            sumPlddt += atom.bfactor;
            atomCount++;
        }
    }

    for (const auto& atom : atoms2) {
        if (interfaceResidues.count(atom.residueId())) {
            sumPlddt += atom.bfactor;
            atomCount++;
        }
    }

    if (atomCount == 0) {
        return 0.0;
    }

    return sumPlddt / atomCount;
}

// ============================================================
// Residue Pairs
// ============================================================

// Get list of contacting residue pairs with their residue names
// pLDDT filtering is atom-level
py::list getResiduePairs(const std::string& pdbText,
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

    if (!chain1.empty() && !chain2.empty()) {
        atoms_set1 = parseAtoms(pdbText, excludeHydrogen, chain1);
        atoms_set2 = parseAtoms(pdbText, excludeHydrogen, chain2);
        interChain = true;
    } else if (!chain.empty()) {
        atoms = parseAtoms(pdbText, excludeHydrogen, chain);
    } else {
        atoms = parseAtoms(pdbText, excludeHydrogen, "");
    }

    // Build residue name map
    std::unordered_map<ResidueId, std::string, ResidueIdHash> residueName;

    auto updateResNames = [&](const std::vector<Atom>& atomList) {
        for (const auto& atom : atomList) {
            ResidueId rid = atom.residueId();
            if (residueName.find(rid) == residueName.end()) {
                residueName[rid] = atom.residueName;
            }
        }
    };

    if (interChain) {
        updateResNames(atoms_set1);
        updateResNames(atoms_set2);
    } else {
        updateResNames(atoms);
    }

    // Collect unique contacting pairs (canonically ordered)
    std::set<std::pair<ResidueId, ResidueId>> contactSet;

    auto addContact = [&](const Atom& a1, const Atom& a2) {
        // Atom-level pLDDT filter
        if (a1.bfactor < minPlddt || a2.bfactor < minPlddt) {
            return;
        }

        double dist = calculateDistance(a1, a2);

        if (dist < cutoff) {
            ResidueId rid1 = a1.residueId();
            ResidueId rid2 = a2.residueId();

            if (rid2 < rid1) std::swap(rid1, rid2);
            contactSet.insert({rid1, rid2});
        }
    };

    if (interChain) {
        for (size_t i = 0; i < atoms_set1.size(); ++i) {
            for (size_t j = 0; j < atoms_set2.size(); ++j) {
                addContact(atoms_set1[i], atoms_set2[j]);
            }
        }
    } else {
        for (size_t i = 0; i < atoms.size(); ++i) {
            for (size_t j = i + 1; j < atoms.size(); ++j) {
                // Skip same residue
                if (atoms[i].residueId() == atoms[j].residueId()) {
                    continue;
                }

                // Check sequence distance (same chain only)
                if (atoms[i].chain == atoms[j].chain) {
                    int seqDist = std::abs(atoms[i].residueNumber - atoms[j].residueNumber);
                    if (seqDist < minSeqDist) {
                        continue;
                    }
                }

                addContact(atoms[i], atoms[j]);
            }
        }
    }

    // Convert to Python list of tuples
    py::list result;
    for (const auto& [rid1, rid2] : contactSet) {
        result.append(py::make_tuple(
            std::string(1, rid1.chain),
            rid1.number,
            residueName[rid1],
            std::string(1, rid2.chain),
            rid2.number,
            residueName[rid2]
        ));
    }

    return result;
}

// ============================================================
// Atom Pairs
// ============================================================

// Get list of contacting atom pairs (atoms not in the same residue)
// pLDDT filtering is atom-level
py::list getAtomPairs(const std::string& pdbText,
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

    if (!chain1.empty() && !chain2.empty()) {
        atoms_set1 = parseAtoms(pdbText, excludeHydrogen, chain1);
        atoms_set2 = parseAtoms(pdbText, excludeHydrogen, chain2);
        interChain = true;
    } else if (!chain.empty()) {
        atoms = parseAtoms(pdbText, excludeHydrogen, chain);
    } else {
        atoms = parseAtoms(pdbText, excludeHydrogen, "");
    }

    py::list result;

    auto addAtomPair = [&](const Atom& a1, const Atom& a2) {
        // Skip if same residue
        if (a1.residueId() == a2.residueId()) {
            return;
        }

        // Atom-level pLDDT filter
        if (a1.bfactor < minPlddt || a2.bfactor < minPlddt) {
            return;
        }

        double dist = calculateDistance(a1, a2);

        if (dist < cutoff) {
            result.append(py::make_tuple(
                std::string(1, a1.chain),
                a1.residueNumber,
                a1.residueName,
                a1.atomName,
                std::string(1, a2.chain),
                a2.residueNumber,
                a2.residueName,
                a2.atomName,
                dist
            ));
        }
    };

    if (interChain) {
        for (size_t i = 0; i < atoms_set1.size(); ++i) {
            for (size_t j = 0; j < atoms_set2.size(); ++j) {
                addAtomPair(atoms_set1[i], atoms_set2[j]);
            }
        }
    } else {
        for (size_t i = 0; i < atoms.size(); ++i) {
            for (size_t j = i + 1; j < atoms.size(); ++j) {
                // Check sequence distance (same chain only)
                if (atoms[i].chain == atoms[j].chain) {
                    int seqDist = std::abs(atoms[i].residueNumber - atoms[j].residueNumber);
                    if (seqDist < minSeqDist) {
                        continue;
                    }
                }

                addAtomPair(atoms[i], atoms[j]);
            }
        }
    }

    return result;
}

// ============================================================
// Contact Order
// ============================================================

// Calculate contact order (average sequence separation of contacts, normalized by length)
// Formula: CO = (1 / (L * N)) * sum(|i - j|) for all contacting pairs (i, j)
py::dict calculateContactOrder(const std::string& pdbText,
                               double cutoff = 4.5,
                               bool excludeHydrogen = true,
                               const std::string& chain = "",
                               int minSeqDist = 1,
                               double minPlddt = 0.0) {
    std::vector<Atom> atoms = parseAtoms(pdbText, excludeHydrogen, chain);

    if (atoms.empty()) {
        throw std::runtime_error("No atoms found matching criteria");
    }

    ResidueIndex ridx = buildResidueIndex(atoms);
    size_t nRes = ridx.residues.size();

    // Build sequential index per chain (for sequence separation)
    std::unordered_map<ResidueId, int, ResidueIdHash> residueSeqIndex;
    std::map<char, int> chainResCount;

    for (const auto& rid : ridx.residues) {
        if (chainResCount.find(rid.chain) == chainResCount.end()) {
            chainResCount[rid.chain] = 0;
        }
        residueSeqIndex[rid] = chainResCount[rid.chain];
        chainResCount[rid.chain]++;
    }

    if (nRes < 2) {
        py::dict result;
        result["contact_order"] = 0.0;
        result["absolute_contact_order"] = 0.0;
        result["num_contacts"] = 0;
        result["num_residues"] = static_cast<int>(nRes);
        result["sum_sequence_separation"] = 0;
        return result;
    }

    // Build residue min-distance matrix
    // Only atom pairs where both atoms pass pLDDT contribute
    std::vector<std::vector<double>> distMatrix(nRes, std::vector<double>(nRes, 1e9));

    for (size_t i = 0; i < nRes; ++i) {
        distMatrix[i][i] = 0.0;
    }

    for (size_t i = 0; i < atoms.size(); ++i) {
        for (size_t j = i + 1; j < atoms.size(); ++j) {
            int resI = ridx.atomToResidue[i];
            int resJ = ridx.atomToResidue[j];

            if (resI != resJ) {
                // Atom-level pLDDT filter
                if (atoms[i].bfactor < minPlddt || atoms[j].bfactor < minPlddt) {
                    continue;
                }

                double dist = calculateDistance(atoms[i], atoms[j]);
                if (dist < distMatrix[resI][resJ]) {
                    distMatrix[resI][resJ] = dist;
                    distMatrix[resJ][resI] = dist;
                }
            }
        }
    }

    // Calculate contact order
    long long sumSeqSep = 0;
    int contactCount = 0;

    for (size_t i = 0; i < nRes; ++i) {
        for (size_t j = i + 1; j < nRes; ++j) {
            // Only consider same-chain contacts for contact order
            if (ridx.residues[i].chain != ridx.residues[j].chain) {
                continue;
            }

            // Get sequence separation using sequential indices
            int seqI = residueSeqIndex[ridx.residues[i]];
            int seqJ = residueSeqIndex[ridx.residues[j]];
            int seqSep = std::abs(seqI - seqJ);

            if (seqSep < minSeqDist) {
                continue;
            }

            if (distMatrix[i][j] < cutoff) {
                sumSeqSep += seqSep;
                contactCount++;
            }
        }
    }

    // Relative CO = (1 / (L * N)) * sum(|i - j|)
    // Absolute CO = (1 / N) * sum(|i - j|)
    double relativeContactOrder = 0.0;
    double absoluteContactOrder = 0.0;

    if (contactCount > 0) {
        absoluteContactOrder = static_cast<double>(sumSeqSep) / contactCount;
        relativeContactOrder = absoluteContactOrder / nRes;
    }

    py::dict result;
    result["contact_order"] = relativeContactOrder;
    result["absolute_contact_order"] = absoluteContactOrder;
    result["num_contacts"] = contactCount;
    result["num_residues"] = static_cast<int>(nRes);
    result["sum_sequence_separation"] = static_cast<int>(sumSeqSep);

    return result;
}

// ============================================================
// Steric Clash Detection
// ============================================================

// Calculate clashscore (ChimeraX-style parameters).
// Overlap = (VDW_A + VDW_B) - distance - hbond_allowance (for donor-acceptor pairs).
// Exclusions: same-residue pairs, and same-chain pairs within min_seq_dist.
py::dict calculateClashScore(const std::string& pdbText,
                             double overlapThreshold = 0.6,
                             double hbondAllowance = 0.4,
                             bool excludeHydrogen = true,
                             const std::string& chain = "",
                             double minPlddt = 0.0,
                             int minSeqDist = 2,
                             double bondViolThreshold = 0.5) {

    std::vector<Atom> atoms = parseAtoms(pdbText, excludeHydrogen, chain);

    if (atoms.empty()) {
        py::dict result;
        result["clashscore"] = 0.0;
        result["num_clashes"] = 0;
        result["num_bond_violations"] = 0;
        result["num_atoms"] = 0;
        return result;
    }

    ResidueIndex ridx = buildResidueIndex(atoms);
    size_t n = atoms.size();
    int clashCount = 0;

    // Precompute VDW radii for all atoms
    std::vector<double> vdwRadii(n);
    for (size_t i = 0; i < n; ++i) {
        vdwRadii[i] = getVdwRadius(atoms[i].element);
    }

    // Max possible VDW radius sum (Se+Se = 3.80)
    double maxRadiusSum = 3.80;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            // Skip same residue
            if (ridx.atomToResidue[i] == ridx.atomToResidue[j]) {
                continue;
            }

            // Sequence distance filter (same chain only)
            if (minSeqDist > 0 && atoms[i].chain == atoms[j].chain) {
                int seqDist = std::abs(atoms[i].residueNumber - atoms[j].residueNumber);
                if (seqDist < minSeqDist) {
                    continue;
                }
            }

            // Atom-level pLDDT filter
            if (atoms[i].bfactor < minPlddt || atoms[j].bfactor < minPlddt) {
                continue;
            }

            double dist = calculateDistance(atoms[i], atoms[j]);

            // Early skip: if distance > maxRadiusSum, no possible clash
            if (dist > maxRadiusSum) {
                continue;
            }

            // Compute overlap with H-bond allowance
            double allowance = 0.0;
            if (hbondAllowance > 0.0 && canHbond(atoms[i].element, atoms[j].element)) {
                allowance = hbondAllowance;
            }

            double overlap = (vdwRadii[i] + vdwRadii[j]) - dist - allowance;

            if (overlap >= overlapThreshold) {
                clashCount++;
            }
        }
    }

    // ----- Covalent backbone bond-length violations -----
    // Detect broken bonds (|distance - ideal| > bondViolThreshold) along the
    // protein and nucleic-acid backbones. Residues are classified by which
    // backbone atoms they contain (name-based), so this is robust to
    // non-standard residue names and atom ordering, handles protein/RNA/DNA
    // complexes in one pass, and stays O(n_atoms).
    struct BBAtoms {
        int N=-1, CA=-1, C=-1, O=-1;                    // protein backbone
        int P=-1, O5=-1, C5=-1, C4=-1, C3=-1, O3=-1;    // nucleic backbone
        bool isProtein() const { return N>=0 && CA>=0 && C>=0; }
        bool isNucleic() const { return C4>=0 && C3>=0; }
    };

    size_t nRes = ridx.residues.size();
    std::vector<BBAtoms> bb(nRes);

    for (size_t k = 0; k < n; ++k) {
        std::string name = atoms[k].atomName;
        for (char& c : name) if (c == '*') c = '\'';    // normalize C5* -> C5'

        BBAtoms& ba = bb[ridx.atomToResidue[k]];
        int idx = static_cast<int>(k);
        if      (name == "N")   ba.N  = idx;
        else if (name == "CA")  ba.CA = idx;
        else if (name == "C")   ba.C  = idx;
        else if (name == "O")   ba.O  = idx;
        else if (name == "P")   ba.P  = idx;
        else if (name == "O5'") ba.O5 = idx;
        else if (name == "C5'") ba.C5 = idx;
        else if (name == "C4'") ba.C4 = idx;
        else if (name == "C3'") ba.C3 = idx;
        else if (name == "O3'") ba.O3 = idx;
    }

    int bondViolations = 0;

    auto checkBond = [&](int ia, int ib, double ideal) {
        if (ia < 0 || ib < 0) return;                   // atom missing: can't judge
        if (atoms[ia].bfactor < minPlddt || atoms[ib].bfactor < minPlddt) return;
        double d = calculateDistance(atoms[ia], atoms[ib]);
        if (std::fabs(d - ideal) > bondViolThreshold) bondViolations++;
    };

    // Intra-residue backbone bonds
    for (size_t r = 0; r < nRes; ++r) {
        if (bb[r].isProtein()) {
            checkBond(bb[r].N,  bb[r].CA, 1.459);
            checkBond(bb[r].CA, bb[r].C,  1.525);
            checkBond(bb[r].C,  bb[r].O,  1.231);
        } else if (bb[r].isNucleic()) {
            checkBond(bb[r].P,  bb[r].O5, 1.593);
            checkBond(bb[r].O5, bb[r].C5, 1.440);
            checkBond(bb[r].C5, bb[r].C4, 1.510);
            checkBond(bb[r].C4, bb[r].C3, 1.524);
            checkBond(bb[r].C3, bb[r].O3, 1.423);
        }
    }

    // Inter-residue linkage bonds, only between consecutive same-chain residues
    // of the same polymer type whose numbers differ by exactly 1 (avoids false
    // positives on sequence gaps and protein/nucleic chain junctions).
    for (size_t r = 0; r + 1 < nRes; ++r) {
        const ResidueId& a = ridx.residues[r];
        const ResidueId& b = ridx.residues[r + 1];
        if (a.chain != b.chain || b.number != a.number + 1) continue;
        if (bb[r].isProtein() && bb[r + 1].isProtein()) {
            checkBond(bb[r].C, bb[r + 1].N, 1.336);     // peptide bond
        } else if (bb[r].isNucleic() && bb[r + 1].isNucleic()) {
            checkBond(bb[r].O3, bb[r + 1].P, 1.607);    // phosphodiester linkage
        }
    }

    // Each broken backbone bond is weighted by (n_atoms * 0.1) so that, after the
    // /n_atoms normalization, it contributes a flat 0.1 to the clashscore regardless
    // of structure size (one break is one break, not size-diluted). A break is far
    // more severe than a steric clash, so it dominates the score; ~10 breaks saturate
    // it. Equivalent to: clashscore = clashCount / n_atoms + 0.1 * bondViolations.
    double clashscore = (static_cast<double>(clashCount)
                         + (static_cast<double>(n) * 0.1) * bondViolations)
                        / static_cast<double>(n);

    py::dict result;
    result["clashscore"] = clashscore;
    result["num_clashes"] = clashCount;
    result["num_bond_violations"] = bondViolations;
    result["num_atoms"] = static_cast<int>(n);

    return result;
}

// ============================================================
// Interface Quality (clashscore + pLDDT)
// ============================================================

// Evaluate the interface between two chain groups.
// 1. Find interface residues (any atom within interfaceCutoff of the other chain)
// 2. Collect all atoms belonging to interface residues from both chains
// 3. Count clashes among those atoms (inter-chain + intra-chain interface)
// 4. Compute average pLDDT of interface atoms
py::dict calculateInterfaceQuality(const std::string& pdbText,
                                    const std::string& chain1,
                                    const std::string& chain2,
                                    double interfaceCutoff = 8.0,
                                    double overlapThreshold = 0.6,
                                    double hbondAllowance = 0.4,
                                    bool excludeHydrogen = true,
                                    double minPlddt = 0.0,
                                    int minSeqDist = 2) {

    if (chain1.empty() || chain2.empty()) {
        throw std::runtime_error("Both chain1 and chain2 must be specified");
    }

    std::vector<Atom> atoms1 = parseAtoms(pdbText, excludeHydrogen, chain1);
    std::vector<Atom> atoms2 = parseAtoms(pdbText, excludeHydrogen, chain2);

    if (atoms1.empty() || atoms2.empty()) {
        py::dict result;
        result["interface_clashscore"] = 0.0;
        result["num_clashes"] = 0;
        result["num_interface_atoms"] = 0;
        result["num_interface_residues"] = 0;
        return result;
    }

    // 1. Find interface residues
    std::set<ResidueId> interfaceResidues;

    for (size_t i = 0; i < atoms1.size(); ++i) {
        for (size_t j = 0; j < atoms2.size(); ++j) {
            double dist = calculateDistance(atoms1[i], atoms2[j]);
            if (dist < interfaceCutoff) {
                interfaceResidues.insert(atoms1[i].residueId());
                interfaceResidues.insert(atoms2[j].residueId());
            }
        }
    }

    if (interfaceResidues.empty()) {
        py::dict result;
        result["interface_clashscore"] = 0.0;
        result["num_clashes"] = 0;
        result["num_interface_atoms"] = 0;
        result["num_interface_residues"] = 0;
        return result;
    }

    // 2. Collect all atoms from interface residues (both chains combined)
    std::vector<Atom> ifaceAtoms;
    for (const auto& atom : atoms1) {
        if (interfaceResidues.count(atom.residueId())) {
            ifaceAtoms.push_back(atom);
        }
    }
    for (const auto& atom : atoms2) {
        if (interfaceResidues.count(atom.residueId())) {
            ifaceAtoms.push_back(atom);
        }
    }

    size_t n = ifaceAtoms.size();
    ResidueIndex ridx = buildResidueIndex(ifaceAtoms);

    // 3. Count clashes among interface atoms
    std::vector<double> vdwRadii(n);
    for (size_t i = 0; i < n; ++i) {
        vdwRadii[i] = getVdwRadius(ifaceAtoms[i].element);
    }

    double maxRadiusSum = 3.80;
    int clashCount = 0;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            // Skip same residue
            if (ridx.atomToResidue[i] == ridx.atomToResidue[j]) {
                continue;
            }

            // Sequence distance filter (same chain only)
            if (minSeqDist > 0 && ifaceAtoms[i].chain == ifaceAtoms[j].chain) {
                int seqDist = std::abs(ifaceAtoms[i].residueNumber - ifaceAtoms[j].residueNumber);
                if (seqDist < minSeqDist) {
                    continue;
                }
            }

            // Atom-level pLDDT filter
            if (ifaceAtoms[i].bfactor < minPlddt || ifaceAtoms[j].bfactor < minPlddt) {
                continue;
            }

            double dist = calculateDistance(ifaceAtoms[i], ifaceAtoms[j]);

            if (dist > maxRadiusSum) {
                continue;
            }

            double allowance = 0.0;
            if (hbondAllowance > 0.0 && canHbond(ifaceAtoms[i].element, ifaceAtoms[j].element)) {
                allowance = hbondAllowance;
            }

            double overlap = (vdwRadii[i] + vdwRadii[j]) - dist - allowance;

            if (overlap >= overlapThreshold) {
                clashCount++;
            }
        }
    }

    double clashscore = static_cast<double>(clashCount) / static_cast<double>(n);

    py::dict result;
    result["interface_clashscore"] = clashscore;
    result["num_clashes"] = clashCount;
    result["num_interface_atoms"] = static_cast<int>(n);
    result["num_interface_residues"] = static_cast<int>(interfaceResidues.size());

    return result;
}

// ============================================================
// Pybind11 Module
// ============================================================

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
          "  chain: Chain filter - comma-separated chain IDs, e.g. 'A,B' for chains A and B (default: all)\n\n"
          "Returns:\n"
          "  NxN NumPy array of distances in Angstroms");

    m.def("interchain_distance_matrix", &calculateInterChainDistanceMatrix,
          py::arg("pdb_text"),
          py::arg("chain1"),
          py::arg("chain2"),
          py::arg("exclude_hydrogen") = true,
          "Calculate atom-level distance matrix between two chain groups.\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  chain1: First chain group, e.g. 'A' or 'A,B'\n"
          "  chain2: Second chain group, e.g. 'C' or 'B,C'\n"
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
          "  chain: Chain filter - comma-separated chain IDs, e.g. 'A,B' (default: all)\n\n"
          "Returns:\n"
          "  MxM NumPy array where M is number of residues");

    m.def("interchain_residue_distance_matrix", &calculateInterChainResidueDistanceMatrix,
          py::arg("pdb_text"),
          py::arg("chain1"),
          py::arg("chain2"),
          py::arg("exclude_hydrogen") = true,
          "Calculate residue-level distance matrix between two chain groups.\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  chain1: First chain group, e.g. 'A' or 'A,B'\n"
          "  chain2: Second chain group, e.g. 'C' or 'B,C'\n"
          "  exclude_hydrogen: Skip hydrogen atoms (default: True)\n\n"
          "Returns:\n"
          "  M1xM2 NumPy array (chain1 residues x chain2 residues)");

    m.def("get_atom_info", &getAtomInfo,
          py::arg("pdb_text"),
          py::arg("exclude_hydrogen") = true,
          py::arg("chain") = "",
          "Get atom and residue information as Python dictionary.\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  exclude_hydrogen: Skip hydrogen atoms (default: True)\n"
          "  chain: Chain filter - comma-separated chain IDs, e.g. 'A,B' (default: all)\n\n"
          "Returns:\n"
          "  Dictionary with keys:\n"
          "    Atom-level:\n"
          "      - atom_names: List of atom names\n"
          "      - residue_names: List of 3-letter residue codes per atom\n"
          "      - residue_numbers: List of residue numbers per atom\n"
          "      - chains: List of chain IDs per atom\n"
          "      - x, y, z: Coordinates\n"
          "      - bfactors: B-factors (pLDDT for AlphaFold)\n"
          "      - n_atoms: Total atom count\n"
          "    Residue-level:\n"
          "      - residues: Dict with chains, numbers, names, n_residues\n"
          "      - sequences: Dict mapping chain ID to one-letter sequence");

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
          "pLDDT filtering is atom-level: both atoms forming a contact must have\n"
          "bfactor >= min_plddt for the contact to count.\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  cutoff: Distance cutoff in Angstroms (default: 4.5)\n"
          "  exclude_hydrogen: Skip hydrogen atoms (default: True)\n"
          "  chain: Calculate contacts within chain(s), e.g. 'A' or 'A,B'\n"
          "  chain1, chain2: Calculate contacts between two chain groups, e.g. 'A' vs 'B,C'\n"
          "  min_plddt: Minimum pLDDT (B-factor) for both atoms in a contact (default: 0.0)\n"
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
          "pLDDT filtering is atom-level: both atoms must have bfactor >= min_plddt.\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  cutoff: Distance cutoff in Angstroms (default: 4.5)\n"
          "  exclude_hydrogen: Skip hydrogen atoms (default: True)\n"
          "  chain: Chain filter - comma-separated chain IDs, e.g. 'A,B' (default: all)\n"
          "  min_seq_dist: Minimum sequence separation (default: 5)\n"
          "  min_plddt: Minimum pLDDT for both atoms in a contact (default: 0.0)\n\n"
          "Returns:\n"
          "  Contact density (float)");

    m.def("interchain_contacts", &calculateInterchainContacts,
          py::arg("pdb_text"),
          py::arg("chain1"),
          py::arg("chain2"),
          py::arg("cutoff") = 4.5,
          py::arg("exclude_hydrogen") = true,
          py::arg("min_plddt") = 0.0,
          "Calculate number of residue contacts between two chain groups.\n\n"
          "pLDDT filtering is atom-level: both atoms must have bfactor >= min_plddt.\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  chain1: First chain group, e.g. 'A' or 'A,B'\n"
          "  chain2: Second chain group, e.g. 'C' or 'B,C'\n"
          "  cutoff: Distance cutoff in Angstroms (default: 4.5)\n"
          "  exclude_hydrogen: Skip hydrogen atoms (default: True)\n"
          "  min_plddt: Minimum pLDDT for both atoms in a contact (default: 0.0)\n\n"
          "Returns:\n"
          "  Number of interchain residue contacts");

    m.def("ligand_contact_density", &calculateLigandContactDensity,
          py::arg("pdb_text"),
          py::arg("polymer_chain"),
          py::arg("ligand_chain"),
          py::arg("cutoff") = 4.5,
          py::arg("exclude_hydrogen") = true,
          py::arg("min_plddt") = 0.0,
          "Calculate ligand contact density (contacts / number of ligand atoms).\n\n"
          "pLDDT filtering is atom-level: both atoms must have bfactor >= min_plddt.\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  polymer_chain: Polymer chain(s), e.g. 'A' or 'A,B' for chains A and B\n"
          "  ligand_chain: Ligand chain(s), e.g. 'B' or 'B,C' for ATP(B) + MG(C)\n"
          "  cutoff: Distance cutoff in Angstroms (default: 4.5)\n"
          "  exclude_hydrogen: Skip hydrogen atoms (default: True)\n"
          "  min_plddt: Minimum pLDDT for both atoms in a contact (default: 0.0)\n\n"
          "Returns:\n"
          "  Contact density: number of contacting residue pairs / number of ligand atoms");

    m.def("interface_plddt", &calculateInterfacePlddt,
          py::arg("pdb_text"),
          py::arg("chain1"),
          py::arg("chain2"),
          py::arg("cutoff") = 4.5,
          py::arg("exclude_hydrogen") = true,
          "Calculate average pLDDT of interface residues.\n\n"
          "Interface residues are those with any atom within cutoff distance\n"
          "of the other chain group. Returns average B-factor of all atoms in these residues.\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  chain1: First chain group, e.g. 'A' or 'A,B'\n"
          "  chain2: Second chain group, e.g. 'C' or 'B,C'\n"
          "  cutoff: Distance cutoff in Angstroms (default: 4.5)\n"
          "  exclude_hydrogen: Skip hydrogen atoms (default: True)\n\n"
          "Returns:\n"
          "  Average pLDDT (B-factor) of interface residues");

    m.def("contact_order", &calculateContactOrder,
          py::arg("pdb_text"),
          py::arg("cutoff") = 4.5,
          py::arg("exclude_hydrogen") = true,
          py::arg("chain") = "",
          py::arg("min_seq_dist") = 1,
          py::arg("min_plddt") = 0.0,
          "Calculate contact order (average sequence separation of contacting residues).\n\n"
          "Formulas:\n"
          "  Relative CO = (1 / (L * N)) * sum(|i - j|)\n"
          "  Absolute CO = (1 / N) * sum(|i - j|)\n"
          "pLDDT filtering is atom-level: both atoms must have bfactor >= min_plddt.\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  cutoff: Distance cutoff in Angstroms (default: 4.5)\n"
          "  exclude_hydrogen: Skip hydrogen atoms (default: True)\n"
          "  chain: Chain filter - comma-separated chain IDs, e.g. 'A,B' (default: all)\n"
          "  min_seq_dist: Minimum sequence separation to count as contact (default: 1)\n"
          "  min_plddt: Minimum pLDDT for both atoms in a contact (default: 0.0)\n\n"
          "Returns:\n"
          "  Dictionary with keys:\n"
          "    - contact_order: Relative contact order (normalized by length)\n"
          "    - absolute_contact_order: Absolute contact order (not normalized)\n"
          "    - num_contacts: Number of contacts found\n"
          "    - num_residues: Number of residues in the chain(s)\n"
          "    - sum_sequence_separation: Total sequence separation of all contacts");

    m.def("get_residue_pairs", &getResiduePairs,
          py::arg("pdb_text"),
          py::arg("cutoff") = 4.5,
          py::arg("exclude_hydrogen") = true,
          py::arg("chain") = "",
          py::arg("chain1") = "",
          py::arg("chain2") = "",
          py::arg("min_plddt") = 0.0,
          py::arg("min_seq_dist") = 0,
          "Get list of contacting residue pairs with residue names.\n\n"
          "pLDDT filtering is atom-level: both atoms must have bfactor >= min_plddt.\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  cutoff: Distance cutoff in Angstroms (default: 4.5)\n"
          "  exclude_hydrogen: Skip hydrogen atoms (default: True)\n"
          "  chain: Chain filter, e.g. 'A' or 'A,B'\n"
          "  chain1, chain2: Calculate contacts between two chain groups\n"
          "  min_plddt: Minimum pLDDT for both atoms in a contact (default: 0.0)\n"
          "  min_seq_dist: Minimum sequence separation for contacts (default: 0)\n\n"
          "Returns:\n"
          "  List of tuples: (chain1, resnum1, resname1, chain2, resnum2, resname2)");

    m.def("get_atom_pairs", &getAtomPairs,
          py::arg("pdb_text"),
          py::arg("cutoff") = 4.5,
          py::arg("exclude_hydrogen") = true,
          py::arg("chain") = "",
          py::arg("chain1") = "",
          py::arg("chain2") = "",
          py::arg("min_plddt") = 0.0,
          py::arg("min_seq_dist") = 0,
          "Get list of contacting atom pairs (atoms not in the same residue).\n\n"
          "pLDDT filtering is atom-level: both atoms must have bfactor >= min_plddt.\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  cutoff: Distance cutoff in Angstroms (default: 4.5)\n"
          "  exclude_hydrogen: Skip hydrogen atoms (default: True)\n"
          "  chain: Chain filter, e.g. 'A' or 'A,B'\n"
          "  chain1, chain2: Calculate contacts between two chain groups\n"
          "  min_plddt: Minimum pLDDT for both atoms in a contact (default: 0.0)\n"
          "  min_seq_dist: Minimum sequence separation for contacts (default: 0)\n\n"
          "Returns:\n"
          "  List of tuples: (chain1, resnum1, resname1, atomname1,\n"
          "                   chain2, resnum2, resname2, atomname2, distance)");

    m.def("clashscore", &calculateClashScore,
          py::arg("pdb_text"),
          py::arg("overlap_threshold") = 0.6,
          py::arg("hbond_allowance") = 0.4,
          py::arg("exclude_hydrogen") = true,
          py::arg("chain") = "",
          py::arg("min_plddt") = 0.0,
          py::arg("min_seq_dist") = 2,
          py::arg("bond_length_violation_threshold") = 0.5,
          "Calculate clashscore = num_clashes / num_atoms + 0.1 * num_bond_violations.\n\n"
          "A steric clash occurs when two non-bonded atoms overlap by more than\n"
          "the threshold. Overlap = (VDW_A + VDW_B) - distance - hbond_allowance.\n"
          "The hbond_allowance is only subtracted for H-bond donor/acceptor pairs (N,O,S).\n"
          "Same-residue pairs are always excluded. Adjacent-residue backbone noise is\n"
          "filtered by min_seq_dist (default: 2 skips seq_dist < 2 on the same chain).\n"
          "pLDDT filtering is atom-level: both atoms must have bfactor >= min_plddt.\n\n"
          "Covalent bond-length violations (broken bonds) are also counted along the\n"
          "protein and nucleic-acid backbones: a bond is a violation when its length\n"
          "deviates from the ideal by more than bond_length_violation_threshold.\n"
          "Residues are classified by backbone-atom presence (name-based), so the check\n"
          "handles protein/RNA/DNA and mixed complexes without residue templates and\n"
          "stays O(n_atoms). Each violation adds a flat 0.1 to the clashscore (size-\n"
          "independent), so a single broken bond dominates clash noise and ~10 saturate it.\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  overlap_threshold: Minimum VDW overlap in Angstroms (default: 0.6, ChimeraX default)\n"
          "  hbond_allowance: Overlap allowance for H-bond pairs in Angstroms (default: 0.4)\n"
          "  exclude_hydrogen: Skip hydrogen atoms (default: True)\n"
          "  chain: Chain filter - comma-separated chain IDs, e.g. 'A,B' (default: all)\n"
          "  min_plddt: Minimum pLDDT for both atoms in a clash or bond (default: 0.0)\n"
          "  min_seq_dist: Minimum sequence separation for same-chain pairs (default: 2)\n"
          "  bond_length_violation_threshold: Max |length - ideal| before a backbone\n"
          "    bond counts as broken, in Angstroms (default: 0.5)\n\n"
          "Returns:\n"
          "  Dictionary with keys:\n"
          "    - clashscore: (num_clashes + num_bond_violations) / num_atoms\n"
          "    - num_clashes: Total number of clashing atom pairs\n"
          "    - num_bond_violations: Total number of broken backbone bonds\n"
          "    - num_atoms: Number of atoms evaluated");

    m.def("interface_quality", &calculateInterfaceQuality,
          py::arg("pdb_text"),
          py::arg("chain1"),
          py::arg("chain2"),
          py::arg("interface_cutoff") = 8.0,
          py::arg("overlap_threshold") = 0.6,
          py::arg("hbond_allowance") = 0.4,
          py::arg("exclude_hydrogen") = true,
          py::arg("min_plddt") = 0.0,
          py::arg("min_seq_dist") = 2,
          "Calculate interface clashscore between two chain groups.\n\n"
          "Identifies interface residues (any atom within interface_cutoff of the other chain),\n"
          "then counts clashes among ALL interface atoms (both inter-chain and intra-chain\n"
          "clashes within interface residues).\n\n"
          "Parameters:\n"
          "  pdb_text: PDB format text\n"
          "  chain1: First chain group, e.g. 'A' or 'A,B'\n"
          "  chain2: Second chain group, e.g. 'C' or 'B,C'\n"
          "  interface_cutoff: Distance cutoff to define interface residues (default: 8.0)\n"
          "  overlap_threshold: Minimum VDW overlap for a clash (default: 0.6)\n"
          "  hbond_allowance: Overlap allowance for H-bond pairs (default: 0.4)\n"
          "  exclude_hydrogen: Skip hydrogen atoms (default: True)\n"
          "  min_plddt: Minimum pLDDT for both atoms in a clash (default: 0.0)\n"
          "  min_seq_dist: Minimum sequence separation for same-chain pairs (default: 2)\n\n"
          "Returns:\n"
          "  Dictionary with keys:\n"
          "    - interface_clashscore: num_clashes / num_interface_atoms\n"
          "    - num_clashes: Total clashing atom pairs in the interface\n"
          "    - num_interface_atoms: Number of atoms in interface residues\n"
          "    - num_interface_residues: Number of interface residues");
}
