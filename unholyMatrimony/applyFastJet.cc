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

int _traverse(PseudoJet root,
              vector<vector<int>>& ints, vector<vector<double>>& doubles,
              int& free_id){
    vector<PseudoJet> stack;
    vector<int> stack_idx;
    // add the first item to the stack
    stack.push_back(root);
    int mother_id = -1;
    
    int daughter1_id;
    int daughter2_id;

    int this_id = free_id++;
    // has form {my_id, global_obs_id, mother_id, daughter1_id, daughter2_id}
    vector<int> root_ints = {this_id, root.user_index(), mother_id, -1, -1};
    ints.push_back(root_ints);
    vector<double> root_doubles = {root.pt(), root.eta(), root.phi_std(), root.e()};
    doubles.push_back(root_doubles);
    stack_idx.push_back(0); // says where the stack item is in terms of ints/floats idx
    int idx_here = stack_idx.back();
    
    while (stack.empty() == false){
        idx_here = stack_idx.back();
        this_id = ints[idx_here][0];
        // check for children
        if (stack.back().has_pieces()){
            daughter1_id = free_id++;
            daughter2_id = free_id++;

            vector<PseudoJet> pieces = stack.back().pieces();
            // remove that vector
            stack.pop_back();
            stack_idx.pop_back();
            // add the new kids
            ints[idx_here][3] = daughter1_id;
            vector<int> daughter1_ints = {daughter1_id, pieces[0].user_index(), this_id, -1, -1};
            ints.push_back(daughter1_ints);
            vector<double> daughter1_doubles = {pieces[0].pt(), pieces[0].eta(), pieces[0].phi_std(), pieces[0].e()};
            doubles.push_back(daughter1_doubles);
            stack_idx.push_back(ints.size() - 1);
            stack.push_back(pieces[0]);

            ints[idx_here][4] = daughter2_id;
            vector<int> daughter2_ints = {daughter2_id, pieces[1].user_index(), this_id, -1, -1};
            ints.push_back(daughter2_ints);
            vector<double> daughter2_doubles = {pieces[1].pt(), pieces[1].eta(), pieces[1].phi_std(), pieces[1].e()};
            doubles.push_back(daughter2_doubles);
            stack_idx.push_back(ints.size() - 1);
            stack.push_back(pieces[1]);
            // add the daughter ids to the parent
        } else {
            // we can remove that item now
            stack.pop_back(); 
            stack_idx.pop_back();
        }
       
    }
    return 0;
}

int print_tree(PseudoJet node, string prefix, bool is_left){
    std::cout << prefix << (is_left ? "|--" : "\\--");
    if( node.has_pieces()){
        std::cout << "\\" << std::endl;
        string new_prefix = prefix + (is_left ? "|  " : "   ");
        vector<PseudoJet> pieces = node.pieces();
        print_tree(pieces[0], new_prefix, true);
        print_tree(pieces[1], new_prefix, false);
    }else{
        std::cout << node.user_index() << std::endl;
    }

}

static void fj(vector<double>& a, // a = flat vector of observations
               vector<vector<int>>& ints, vector<vector<double>>& doubles,
               vector< double >& masses,  // per jet results
               vector< double >& pts,  // per jet results
               double R=1.0, int algorithm=0) {
    // Extract particles from array
    vector<fastjet::PseudoJet> particles;

    string n_output_name = "fastjet_njets.csv";
    std::ofstream n_file(n_output_name, std::ios_base::app);
    string sep = " ";
    for (unsigned int i = 0; i < a.size(); i += 5) {
        //                               px    py      pz      e
        fastjet::PseudoJet p = PseudoJet(a[i+1], a[i+2], a[i+3], a[i+4]);
        // this is the global_obs_id
        p.set_user_index((int) a[i]);
        std::cout << "Particle " << i/5 << " pt=" << p.pt() << " eta=" << p.eta() << " phi=" << p.phi_std() << " e=" << p.e() << std::endl;
        n_file << p.px() << sep
             << p.py() << sep
             << p.pz() << sep
             << p.e() << sep
             << p.pt()  << sep
             << p.eta()  << sep
             << p.phi_std() << sep
             << p.m() << sep;
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
    int free_id = 0;

    for (unsigned int j = 0; j < jets.size(); j++) {
        /*
        std::cout << "~~~~~~~~~~~~~~~" << std::endl;
        std::cout << "Jet number " << j << std::endl;
        print_tree(jets[j], "", false);
        std::cout << "~~~~~~~~~~~~~~~" << std::endl;
        */
        vector<vector<int>> ints_here;
        vector<vector<double>> doubles_here;
        int success = _traverse(jets[j], ints_here, doubles_here, free_id);
        ints.insert(ints.end(), ints_here.begin(), ints_here.end());
        doubles.insert(doubles.end(), doubles_here.begin(), doubles_here.end());
        masses.push_back(jets[j].m());
        pts.push_back(jets[j].pt());
    }
    n_file << jets.size() << std::endl;
    std::cout << "\t\tNumber jets = " << jets.size() << "\n\n";

}


int main(int argc, char * argv[]) {
    if(argc != 4){
        std::cout << "The arguments should be "
                  << "<folder name> "
                  << "<deltaR> "
                  << "<algorithm_num>" << std::endl;
        return 1;
    }
    // the argument shoulf be the directory of the files
    std::string dir_name(argv[1]);
    if(dir_name.back() != '/'){
        dir_name = dir_name + "/";
    }
    std::string input_name = dir_name + "summary_observables.csv";
    std::ifstream file(input_name);
    CSVRow row;
    vector<double> a;
    // first row is a header
    file >> row;
    while(file >> row) {
        // these values come from the observabels/summary file
        a.push_back(std::stod(row[0]));  //global_obs_id
        a.push_back(std::stod(row[5]));  //px
        a.push_back(std::stod(row[6]));  //py
        a.push_back(std::stod(row[7]));  //pz
        a.push_back(std::stod(row[4]));  //e
    }
   vector< vector<int> > ints;  // for geometry results
   vector< vector<double> > doubles; //  for kinematics results
   vector< double > masses;  // per jet results
   vector< double > pts;  // per jet results

   // run the algorithm
   double R = atof(argv[2]);
   int algorithm = atoi(argv[3]);
   string algorithm_name = algorithm == 0 ? "kt_algorithm" : (algorithm == 1 ? "antikt_algorithm" : "cambridge_algorithm");
   fj(a, ints, doubles , masses, pts, R, algorithm);

   //std::cout << "Read " << ints.size()/4 << " particles\n";

   //prepare to write
   string sep = " ";

   string i_output_name = dir_name + "fastjet_ints.csv";
   std::ofstream int_file(i_output_name);
   int_file << "# " << "deltaR=" << R << sep
                    << algorithm_name << sep
                    << "Columns;" << sep
                    << "psudojet_id" << sep
                    << "global_obs_id" << sep
                    << "mother_id" << sep
                    << "daughter1_id" << sep
                    << "daughter2_id"
                    << std::endl;
   string d_output_name = dir_name + "fastjet_doubles.csv";
   std::ofstream double_file(d_output_name);
   double_file << "# " << "pt" << sep
                       << "eta" << sep
                       << "phi" << sep
                       << "e"
                       << std::endl;

   // Write out
   for(int i =0; i < ints.size(); i++){
       int_file << ints[i][0] << sep 
                << ints[i][1] << sep
                << ints[i][2] << sep
                << ints[i][3] << sep
                << ints[i][4]
                << std::endl;
       double_file << doubles[i][0] << sep 
                   << doubles[i][1] << sep
                   << doubles[i][2] << sep
                   << doubles[i][3]
                   << std::endl;
   }
}
