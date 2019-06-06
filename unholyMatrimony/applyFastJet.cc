// compile me with 
// g++ applyFastJet.cc -o applyFastJet `/usr/local/bin/fastjet-config --cxxflags --libs --plugins`
// g++ applyFastJet.cc -o applyFastJet -I/usr/local/include -Wl,-rpath,/usr/local/lib -lm -L/usr/local/lib -lfastjettools -lfastjet -lfastjetplugins -lsiscone_spherical -lsiscone
//
#include <iostream>
#include "/home/henry/Programs/fastjet/include/fastjet/ClusterSequence.hh"
#include <fstream>

using namespace std;
using namespace fastjet;

// For getting input (why is this so haaard :p) ~~~~~~~~
class CSVRow
{
    public:
        std::string const& operator[](std::size_t index) const
        {
            return m_data[index];
        }
        std::size_t size() const
        {
            return m_data.size();
        }
        void readNextRow(std::istream& str)
        {
            std::string         line;
            std::getline(str, line);

            std::stringstream   lineStream(line);
            std::string         cell;

            m_data.clear();
            while(std::getline(lineStream, cell, ' '))
            {
                m_data.push_back(cell);
            }
            // This checks for a trailing comma with no data after it.
            if (!lineStream && cell.empty())
            {
                // If there was a trailing comma then add an empty element.
                m_data.push_back("");
            }
        }
    private:
        std::vector<std::string>    m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data)
{
    data.readNextRow(str);
    return str;
}   
// ~~~~~~~~~~~~~~~~~~~~~~`

void _traverse_rec(PseudoJet root, int parent_id, bool is_left,
                  vector<int>& tree, vector<double>& content){
    // id for the root
    int id = tree.size() / 2;
    // the divide by 2 come from the fact that each new node
    // adds 2 children to the tree
    // so the id of the nth node is tree.size()/2

    // if this isn't the root
    if (parent_id >= 0) {
        if (is_left) {
            tree[2 * parent_id] = id;
        } else {
            tree[2 * parent_id + 1] = id;
        }
    }

    // there are two pottential children here, give them each space on the tree
    tree.push_back(-1);
    tree.push_back(-1);
    content.push_back(root.px());
    content.push_back(root.py());
    content.push_back(root.pz());
    content.push_back(root.e());
    content.push_back(root.user_index());  // remove this for jet studies  << only original comment

    // Doc for has_pieces
      /// returns true if a jet has pieces
      ///
      /// By default a single particle or a jet coming from a
      /// ClusterSequence have no pieces and this methos will return false.
      ///
      /// In practice, this is equivalent to have an structure of type
      /// CompositeJetStructure.
    if (root.has_pieces()) {
        vector<PseudoJet> pieces = root.pieces();
        // this will decend the left edge until root.has_pieces == False
        //
        _traverse_rec(pieces[0], id, true, tree, content);
        _traverse_rec(pieces[1], id, false, tree, content);
    }
}

pair< vector<int>, vector<double> > _traverse(PseudoJet root){
    vector<int> tree;
    vector<double> content;
    //        entry-pt parent-id is_left storage storage
    _traverse_rec(root, -1, false, tree, content);
    return make_pair(tree, content);
}

static void fj(vector<double>& a, // a = flat vector of observations
               vector< vector<int> >& trees,  // for geometry results
               vector< vector<double> >& contents, //  for kinematics results
               vector< double >& masses,  // per jet results
               vector< double >& pts,  // per jet results
               double R=1.0, int algorithm=0) {
    // Extract particles from array
    vector<fastjet::PseudoJet> particles;

    for (unsigned int i = 0; i < a.size(); i += 4) {
        //                               px    py      pz      e
        fastjet::PseudoJet p = PseudoJet(a[i], a[i+1], a[i+2], a[i+3]);
        p.set_user_index((int) i / 4);
        particles.push_back(p);
    }

    // Cluster
    JetDefinition def(algorithm == 0 ? kt_algorithm : (algorithm == 1 ? antikt_algorithm : cambridge_algorithm), R);
    ClusterSequence seq(particles, def);
    // must assume that inclusive_jets in some sense means "produce jets with CompositeJetStructure"
    // Doc for inclusive_jets
      /// return a vector of all jets (in the sense of the inclusive
      /// algorithm) with pt >= ptmin. Time taken should be of the order
      /// of the number of jets returned.
    vector<PseudoJet> jets = sorted_by_pt(seq.inclusive_jets());
    // Doc for compositeJetStructure
        /// \class CompositeJetStructure
        /// The structure for a jet made of pieces
        ///
        /// This stores the vector of the pieces that make the jet and provide
        /// the methods to access them

    // Store results
    for (unsigned int j = 0; j < jets.size(); j++) {
        pair< vector<int>, vector<double> > p = _traverse(jets[j]);
        trees.push_back(p.first);
        contents.push_back(p.second);
        masses.push_back(jets[j].m());
        pts.push_back(jets[j].pt());
    }
}


int main(int argc, char * argv[]) {
    // the arguments should be the name of the csv containing the observations
    char * file_name(argv[1]);
    std::ifstream file(file_name);
    CSVRow row;
    vector<double> a;
    // first row is a header
    file >> row;
    while(file >> row) {
        // these values come from the observabels/summary file
        a.push_back(std::stod(row[5]));  //px
        a.push_back(std::stod(row[6]));  //py
        a.push_back(std::stod(row[7]));  //pz
        a.push_back(std::stod(row[4]));  //e
    }
   vector< vector<int> > trees;  // for geometry results
   vector< vector<double> > contents; //  for kinematics results
   vector< double > masses;  // per jet results
   vector< double > pts;  // per jet results

   // run the algorithm
   fj(a, trees, contents, masses, pts);

   // This format is horrible
}
