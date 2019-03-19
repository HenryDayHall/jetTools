/*
   This macro shows how to access the particle-level reference for reconstructed objects.
   It is also shown how to loop over the jet constituents.

// To run in root
//  $ cd ~/Programs/madgraph/Delphes
//  $ root -b
//  [0] gSystem->Load("/usr/lib/x86_64-linux-gnu/libsqlite3.so")
//  [1] .x examples/to_sql.C
*/

// these includes are for the SQL stuff ~~~~~~~~~~~~~~~~~~~~
#include <stdio.h>
#include <stdlib.h>
#include <sqlite3.h> 
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#include "external/ExRootAnalysis/ExRootResult.h"
#else
    class ExRootTreeReader;
    class ExRootResult;
#endif

    using std::string;

// Helper functions for SQLite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
static int callback(void *NotUsed, int argc, char **argv, char **azColName) {
    int i;
    for(i = 0; i<argc; i++) {
        printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
    }
    printf("\n");
    return 0;
}

void check_execution(int rc, char *zErrMsg){
    if( rc != SQLITE_OK ){
        fprintf(stderr, "SQL error: %s\n", zErrMsg);
        sqlite3_free(zErrMsg);
    } else {
        //fprintf(stdout, "Command executed sucessfully\n");
    }
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


string string_from_TLVec(TLorentzVector indexable)
{
    int length = 4;
    std::stringstream stream;
    stream << "[";
    for(int i = 0; i < length-1; i++)
    {
        stream << indexable[i] << ", ";
    }
    stream << indexable[length-1] << "]";
    string out = stream.str();
    return out;
}


void PrintParticle(GenParticle * particle, int verbosity, string * out) {
    std::stringstream stream;
    if(verbosity < 0){
        stream << particle->PID << ", ";
        stream << particle->Status << ", ";
        stream << particle->IsPU << ", ";
        stream << particle->Charge << ", ";
        stream << particle->Mass << ", ";
        stream << particle->E << ", ";
        stream << particle->Px << ", ";
        stream << particle->Py << ", ";
        stream << particle->Pz << ", ";
        stream << particle->P << ", ";
        stream << particle->PT << ", ";
        stream << particle->Eta << ", ";
        stream << particle->Phi << ", ";
        stream << particle->Rapidity << ", ";
        stream << particle->CtgTheta << ", ";
        stream << particle->D0 << ", ";
        stream << particle->DZ << ", ";
        stream << particle->T << ", ";
        stream << particle->X << ", ";
        stream << particle->Y << ", ";
        stream << particle->Z;
    }else if( verbosity == 0) {
        stream << "PID = " << particle->PID << "; ";
        stream << "E = " << particle->E << "; ";
        stream << "particle->P4() = " << string_from_TLVec(particle->P4()) << endl;
    } else if (verbosity == 1) {
        stream << "PID = " << particle->PID << "; ";
        stream << "Status = " << particle->Status << "; ";
        stream << "IsPU = " << particle->IsPU << "; ";
        stream << "M1 = " << particle->M1 << "; ";
        stream << "M2 = " << particle->M2 << "; ";
        stream << "D1 = " << particle->D1 << "; ";
        stream << "D2 = " << particle->D2 << "; ";
        stream << "Charge = " << particle->Charge << "; ";
        stream << "Mass = " << particle->Mass << "; ";
        stream << "E = " << particle->E << "; ";
        stream << "Px = " << particle->Px << "; ";
        stream << "Py = " << particle->Py << "; ";
        stream << "Pz = " << particle->Pz << "; ";
        stream << "particle->P4() = " << string_from_TLVec(particle->P4()) << endl;
    } else {
        stream << "particle HEP ID number | hepevt.idhep[number] " << particle->PID << endl;
        stream << "particle status | hepevt.isthep[number] " << particle->Status << endl;
        stream << "0 or 1 for particles from pile-up interactions " << particle->IsPU << endl;
        stream << "particle 1st mother | hepevt.jmohep[number][0] - 1 " << particle->M1 << endl;
        stream << "particle 2nd mother | hepevt.jmohep[number][1] - 1 " << particle->M2 << endl;
        stream << "particle 1st daughter | hepevt.jdahep[number][0] - 1 " << particle->D1 << endl;
        stream << "particle last daughter | hepevt.jdahep[number][1] - 1 " << particle->D2 << endl;
        stream << "particle charge " << particle->Charge << endl;
        stream << "particle mass " << particle->Mass << endl;
        stream << "particle energy | hepevt.phep[number][3] " << particle->E << endl;
        stream << "particle momentum vector (x component) | hepevt.phep[number][0] " << particle->Px << endl;
        stream << "particle momentum vector (y component) | hepevt.phep[number][1] " << particle->Py << endl;
        stream << "particle momentum vector (z component) | hepevt.phep[number][2] " << particle->Pz << endl;
        stream << "particle momentum " << particle->P << endl;
        stream << "particle transverse momentum " << particle->PT << endl;
        stream << "particle pseudorapidity " << particle->Eta << endl;
        stream << "particle azimuthal angle " << particle->Phi << endl;
        stream << "particle rapidity " << particle->Rapidity << endl;
        stream << "particle cotangent of theta " << particle->CtgTheta << endl;
        stream << "particle transverse impact parameter " << particle->D0 << endl;
        stream << "particle longitudinal impact parameter " << particle->DZ << endl;
        stream << "particle vertex position (t component) | hepevt.vhep[number][3] " << particle->T << endl;
        stream << "particle vertex position (x component) | hepevt.vhep[number][0] " << particle->X << endl;
        stream << "particle vertex position (y component) | hepevt.vhep[number][1] " << particle->Y << endl;
        stream << "particle vertex position (z component) | hepevt.vhep[number][2] " << particle->Z << endl;
        stream << "particle->P4() = " << string_from_TLVec(particle->P4()) << endl;
    }

    *out = stream.str();
}

void PrintParticle(GenParticle * particle, int verbosity = 0) {
    string * out = new string();
    PrintParticle(particle, verbosity, out);
    cout << *out;
    delete out;
}

void PrintTrack(Track * track, int verbosity, string *out) {
    std::stringstream stream;
    if(verbosity < 0){
        stream << track->PID << ", ";
        stream << track->Charge << ", ";
        stream << track->P << ", ";
        stream << track->PT << ", ";
        stream << track->Eta << ", ";
        stream << track->Phi << ", ";
        stream << track->CtgTheta << ", ";
        stream << track->EtaOuter << ", ";
        stream << track->PhiOuter << ", ";
        stream << track->T << ", ";
        stream << track->X << ", ";
        stream << track->Y << ", ";
        stream << track->Z << ", ";
        stream << track->TOuter << ", ";
        stream << track->XOuter << ", ";
        stream << track->YOuter << ", ";
        stream << track->ZOuter << ", ";
        stream << track->Xd << ", ";
        stream << track->Yd << ", ";
        stream << track->Zd << ", ";
        stream << track->L << ", ";
        stream << track->D0 << ", ";
        stream << track->DZ << ", ";
        stream << track->ErrorP << ", ";
        stream << track->ErrorPT << ", ";
        stream << track->ErrorPhi << ", ";
        stream << track->ErrorCtgTheta << ", ";
        stream << track->ErrorT << ", ";
        stream << track->ErrorD0 << ", ";
        stream << track->ErrorDZ;
    } else if(verbosity == 0) {
        stream << "PID = " << track->PID << "; track->P4() = " << string_from_TLVec(track->P4()) << endl;
    } else if(verbosity == 1) {
        stream << "PID = " << track->PID << "; ";
        stream << "Charge = " << track->Charge << "; ";
        stream << "P = " << track->P << "; ";
        stream << "PT = " << track->PT << "; ";
        stream << "Eta = " << track->Eta << "; ";
        stream << "Phi = " << track->Phi << "; ";
        stream << "Xd = " << track->Xd << "; ";
        stream << "Yd = " << track->Yd << "; ";
        stream << "Zd = " << track->Zd << "; ";
        stream << "L = " << track->L << "; ";
        stream << "track->P4() = " << string_from_TLVec(track->P4()) << endl;
    } else {
        stream << "HEP ID number " << track->PID << endl;
        stream << "track charge " << track->Charge << endl;
        stream << "track momentum " << track->P << endl;
        stream << "track transverse momentum " << track->PT << endl;
        stream << "track pseudorapidity " << track->Eta << endl;
        stream << "track azimuthal angle " << track->Phi << endl;
        stream << "track cotangent of theta " << track->CtgTheta << endl;
        stream << "track pseudorapidity at the tracker edge " << track->EtaOuter << endl;
        stream << "track azimuthal angle at the tracker edge " << track->PhiOuter << endl;
        stream << "track vertex position (t component) " << track->T << endl;
        stream << "track vertex position (x component) " << track->X << endl;
        stream << "track vertex position (y component) " << track->Y << endl;
        stream << "track vertex position (z component) " << track->Z << endl;
        stream << "track position (t component) at the tracker edge " << track->TOuter << endl;
        stream << "track position (x component) at the tracker edge " << track->XOuter << endl;
        stream << "track position (y component) at the tracker edge " << track->YOuter << endl;
        stream << "track position (z component) at the tracker edge " << track->ZOuter << endl;
        stream << "X coordinate of point of closest approach to vertex " << track->Xd << endl;
        stream << "Y coordinate of point of closest approach to vertex " << track->Yd << endl;
        stream << "Z coordinate of point of closest approach to vertex " << track->Zd << endl;
        stream << "track path length " << track->L << endl;
        stream << "track transverse impact parameter " << track->D0 << endl;
        stream << "track longitudinal impact parameter " << track->DZ << endl;
        stream << "track momentum error " << track->ErrorP << endl;
        stream << "track transverse momentum error " << track->ErrorPT << endl;
        stream << "track azimuthal angle error " << track->ErrorPhi << endl;
        stream << "track cotangent of theta error " << track->ErrorCtgTheta << endl;
        stream << "time measurement error " << track->ErrorT << endl;
        stream << "track transverse impact parameter error " << track->ErrorD0 << endl;
        stream << "track longitudinal impact parameter error " << track->ErrorDZ << endl;
        stream << "track->P4() = " << string_from_TLVec(track->P4()) << endl;

        stream << "Gen particle; " << endl;
        GenParticle * particle = (GenParticle *)track->Particle.GetObject(); 
        string * particle_print;
        PrintParticle(particle, verbosity, particle_print);
        stream << particle_print;
    }
    *out = stream.str();
}

void PrintTrack(Track *track, int verbosity = 0) {
    string * out = new string();
    PrintTrack(track, verbosity, out);
    cout << *out;
    delete out;
}


void PrintTower(Tower * tower, int verbosity, string *out) {
    std::stringstream stream;
    if(verbosity < 0){
        stream << tower->ET << ", ";
        stream << tower->Eta << ", ";
        stream << tower->Phi << ", ";
        stream << tower->E << ", ";
        stream << tower->T << ", ";
        stream << tower->NTimeHits << ", ";
        stream << tower->Eem << ", ";
        stream << tower->Ehad << ", ";
        stream << tower->Edges[0] << ", ";
        stream << tower->Edges[1] << ", ";
        stream << tower->Edges[2] << ", ";
        stream << tower->Edges[3];
    }else if(verbosity == 1) {
        stream << "ET = " << tower->ET << "; ";
        stream << "Eta = " << tower->Eta << "; ";
        stream << "Phi = " << tower->Phi << "; ";
        stream << "tower->P4() = " << string_from_TLVec(tower->P4()) << endl;
    } else if (verbosity == 1) {
        stream << "ET = " << tower->ET << "; ";
        stream << "Eta = " << tower->Eta << "; ";
        stream << "Phi = " << tower->Phi << "; ";
        stream << "E = " << tower->E << "; ";
        stream << "T = " << tower->T << "; ";
        stream << "NTimeHits = " << tower->NTimeHits << "; ";
        stream << "Eem = " << tower->Eem << "; ";
        stream << "Ehad = " << tower->Ehad << "; ";
        stream << "tower->P4() = " << string_from_TLVec(tower->P4()) << endl;
    } else {
        stream << "calorimeter tower transverse energy " << tower->ET << endl;
        stream << "calorimeter tower pseudorapidity " << tower->Eta << endl;
        stream << "calorimeter tower azimuthal angle " << tower->Phi << endl;
        stream << "calorimeter tower energy " << tower->E << endl;
        stream << "ecal deposit time, averaged by sqrt(EM energy) over all particles, not smeared " << tower->T << endl;
        stream << "number of hits contributing to time measurement " << tower->NTimeHits << endl;
        stream << "calorimeter tower electromagnetic energy " << tower->Eem << endl;
        stream << "calorimeter tower hadronic energy " << tower->Ehad << endl;
        stream << "tower->P4() = " << string_from_TLVec(tower->P4()) << endl;
    }
    *out = stream.str();
}


void PrintTower(Tower *tower, int verbosity = 0) {
    string * out = new string();
    PrintTower(tower, verbosity, out);
    cout << *out;
    delete out;
}

//------------------------------------------------------------------------------
void FindAndReplaceAll(std::string & data, std::string toSearch, std::string replaceStr)
{
	// Get the first occurrence
	size_t pos = data.find(toSearch);
 
	// Repeat till end is reached
	while( pos != std::string::npos)
	{
		// Replace this occurrence of Sub String
		data.replace(pos, toSearch.size(), replaceStr);
		// Get the next occurrence from the current position
		pos =data.find(toSearch, pos + replaceStr.size());
	}
}

void ConvertFilename(string &fileName, const string newExtention = ".db"){
    string::size_type extPos = fileName.rfind('.', fileName.length());
    if(extPos != string::npos){ // there is a dot int he string
        fileName = fileName.substr(0, extPos);
    }
    fileName += newExtention;
}

sqlite3 * CreateDatabase(const char *rootFile){
    sqlite3 *db;

    char *zErrMsg = 0;
    int rc;
    const char *sql;
    const char *data = "Callback function called";

    string newName(rootFile);
    ConvertFilename(newName);
	FindAndReplaceAll(newName, "~/", "/home/henry/");

    /* Open database */
    rc = sqlite3_open(newName.c_str(), &db); // wants  const char * 

    if( rc ) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        return(0);
    } else {
        fprintf(stderr, "Opened database successfully\n");
    }

    /* Create the table. Create SQL statement */
    sql = "CREATE TABLE GenParticles("
          "ID INTEGER PRIMARY KEY NOT NULL," // Primary key for the particle
          "MCPID INT," // particle HEP ID number | hepevt.idhep[number]
          "Status INT," // particle status | hepevt.isthep[number]
          "IsPU INT," // 0 or 1 for particles from pile-up interactions
          "M1 INT," // particle 1st mother | hepevt.jmohep[number][0] - 1
          "M2 INT," // particle 2nd mother | hepevt.jmohep[number][1] - 1
          "D1 INT," // particle 1st daughter | hepevt.jdahep[number][0] - 1
          "D2 INT," // particle 2nd daughter | hepevt.jdahep[number][1] - 1
          "Charge INT," // particle charge
          "Mass REAL," // particle mass
          "E REAL," // particle energy | hepevt.phep[number][3]
          "Px REAL," // particle momentum vector (x component) | hepevt.phep[number][0]
          "Py REAL," // particle momentum vector (y component) | hepevt.phep[number][1]
          "Pz REAL," // particle momentum vector (z component) | hepevt.phep[number][2]
          "P REAL," // particle momentum
          "PT REAL," // particle transverse momentum
          "Eta REAL," // particle pseudorapidity
          "Phi REAL," // particle azimuthal angle
          "Rapidity REAL," // particle rapidity
          "CtgTheta REAL," // particle cotangent of theta
          "D0 REAL," // particle transverse impact parameter
          "DZ REAL," // particle longitudinal impact parameter
          "T REAL," // particle vertex position (t component) | hepevt.vhep[number][3]
          "X REAL," // particle vertex position (x component) | hepevt.vhep[number][0]
          "Y REAL," // particle vertex position (y component) | hepevt.vhep[number][1]
          "Z REAL," // particle vertex position (z component) | hepevt.vhep[number][2]
          "FOREIGN KEY(M1) REFERENCES GenParticles(ID),"
          "FOREIGN KEY(M2) REFERENCES GenParticles(ID),"
          "FOREIGN KEY(D1) REFERENCES GenParticles(ID),"
          "FOREIGN KEY(D2) REFERENCES GenParticles(ID));";

    /* Execute SQL statement */
    rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);

    check_execution(rc, zErrMsg);

    /* Create the table. Create SQL statement */
    sql = "CREATE TABLE Tracks("
          "ID INTEGER PRIMARY KEY NOT NULL," // Primary key for the track
          "MCPID INTEGER," // HEP ID number
          "Charge INTEGER ," // track charge
          "P REAL," // track momentum
          "PT REAL," // track transverse momentum
          "Eta REAL," // track pseudorapidity
          "Phi REAL," // track azimuthal angle
          "CtgTheta REAL ," // track cotangent of theta
          "EtaOuter REAL ," // track pseudorapidity at the tracker edge
          "PhiOuter REAL ," // track azimuthal angle at the tracker edge
          "T REAL," // track vertex position (t component)
          "X REAL," // track vertex position (x component)
          "Y REAL," // track vertex position (y component)
          "Z REAL," // track vertex position (z component)
          "TOuter REAL," // track position (t component) at the tracker edge
          "XOuter REAL," // track position (x component) at the tracker edge
          "YOuter REAL," // track position (y component) at the tracker edge
          "ZOuter REAL," // track position (z component) at the tracker edge
          "Xd REAL," // X coordinate of point of closest approach to vertex
          "Yd REAL," // Y coordinate of point of closest approach to vertex
          "Zd REAL," // Z coordinate of point of closest approach to vertex
          "L REAL," // track path length
          "D0 REAL," // track transverse impact parameter
          "DZ REAL," // track longitudinal impact parameter
          "ErrorP REAL," // track momentum error
          "ErrorPT REAL," // track transverse momentum error
          "ErrorPhi REAL ," // track azimuthal angle error
          "ErrorCtgTheta REAL," // track cotangent of theta error
          "ErrorT REAL," // time measurement error
          "ErrorD0 REAL," // track transverse impact parameter error
          "ErrorDZ REAL," // track longitudinal impact parameter error
          "Particle INTEGER," // reference to generated particle
          "VertexIndex INTEGER," // reference to vertex TODO should this be a forign key??
          "FOREIGN KEY(Particle) REFERENCES GenParticles(ID));";

    /* Execute SQL statement */
    rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);

    check_execution(rc, zErrMsg);

    /* Create the table. Create SQL statement */
    sql = "CREATE TABLE Towers("
          "ID INTEGER PRIMARY KEY NOT NULL," // Primary key for the tower
          "ET REAL, " // calorimeter tower transverse energy
          "Eta REAL,"  // calorimeter tower pseudorapidity
          "Phi REAL,"  // calorimeter tower azimuthal angle
          "E REAL,"  // calorimeter tower energy
          "T REAL,"  // ecal deposit time, averaged by sqrt(EM energy) over all particles, not smeared
          "NTimeHits INTEGER,"  // number of hits contributing to time measurement
          "Eem REAL,"  // calorimeter tower electromagnetic energy
          "Ehad REAL,"  // calorimeter tower hadronic energy
          "Edge1 REAL," // calorimeter tower edges
          "Edge2 REAL," // calorimeter tower edges
          "Edge3 REAL," // calorimeter tower edges
          "Edge4 REAL);" // calorimeter tower edges
          "CREATE TABLE TowerLinks("  //gotta do this seperatly as it is one to many
          "ID INTEGER PRIMARY KEY NOT NULL," // Primary key for the tower link
          "Particle INTEGER," // reference to generated particle
          "Tower INTEGER," // reference to tower
          "FOREIGN KEY(Tower) REFERENCES Towers(ID),"
          "FOREIGN KEY(Particle) REFERENCES GenParticles(ID));";

    /* Execute SQL statement */
    rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);

    check_execution(rc, zErrMsg);
    return db;
}


void AddParticle(GenParticle * particle, int SQLkey, sqlite3 * db){
    //Adds the particle, but not the relations to other particles
    char *zErrMsg = 0;
    int rc;
    const char *sql;

    /* Insert the particle, create a statment*/
    std::stringstream glue;
    glue << "INSERT INTO GenParticles(";
    glue << "ID, MCPID, Status, IsPU, Charge, Mass, E, Px, Py, Pz, P, PT, Eta, Phi, Rapidity, CtgTheta, D0, DZ, T, X, Y, Z)";
    glue << " VALUES (" << SQLkey << ", ";
    string *particle_print = new string();
    PrintParticle(particle, -1, particle_print);
    glue << *particle_print << ");";
    const string tmp = glue.str();
    sql = tmp.c_str();

    /* Execute SQL statement */
    rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);

    check_execution(rc, zErrMsg);
    delete particle_print;
}


void AddTrack(Track *track, std::map<string, int> *particle_to_SQLkey, int SQLkey, sqlite3 * db){
    char *zErrMsg = 0;
    int rc;
    const char *sql;

    /* Insert the particle, create a statment*/
    std::stringstream glue;
    glue << "INSERT INTO Tracks(";
    glue << "ID, MCPID, Charge, P, PT, Eta, Phi, CtgTheta, EtaOuter, PhiOuter, ";
    glue << "T, X, Y, Z, TOuter, XOuter, YOuter, ZOuter, Xd, Yd, Zd, L, D0, DZ, ";
    glue << "ErrorP, ErrorPT, ErrorPhi, ErrorCtgTheta, ErrorT, ErrorD0, ErrorDZ, ";
    glue << "Particle)";
    glue << " VALUES (" << SQLkey << ", ";
    string *track_print = new string();
    PrintTrack(track, -1, track_print);
    string *particle_print = new string();
    GenParticle *particle = (GenParticle *)track->Particle.GetObject();
    PrintParticle(particle, -1, particle_print);
    int particle_foreignKey = (*particle_to_SQLkey)[*particle_print];
    glue << *track_print << ", " << particle_foreignKey << ");";
    const string tmp = glue.str();
    sql = tmp.c_str();

    /* Execute SQL statement */
    rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);

    check_execution(rc, zErrMsg);
    delete track_print;
}


void AddTower(Tower *tower, std::map<string, int> *particle_to_SQLkey, int SQLkey, sqlite3 * db){
    char *zErrMsg = 0;
    int rc;
    const char *sql;

    /* Insert the tower, create a statment*/
    std::stringstream glue;
    glue << "INSERT INTO Towers(";
    glue << "ID, ET, Eta, Phi, E, T, NTimeHits, Eem, Ehad, Edge1, Edge2, Edge3, Edge4)";
    glue << " VALUES (" << SQLkey << ", ";
    string *tower_print = new string();
    PrintTower(tower, -1, tower_print);
    glue << *tower_print << ");";
    const string tmp = glue.str();
    sql = tmp.c_str();

    /* Execute SQL statement */
    rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);

    check_execution(rc, zErrMsg);
    delete tower_print;
    // Now add links to particles that make this tower
    for(int j = 0; j < tower->Particles.GetEntriesFast(); ++j) {
        GenParticle *particle = (GenParticle*) tower->Particles.At(j);
        string *particle_print = new string();
        PrintParticle(particle, -1, particle_print);
        int particle_foreignKey = (*particle_to_SQLkey)[*particle_print];
        std::stringstream glue;
        //int link_key = (j + 1)*SQLkey + j;
        glue << "INSERT INTO TowerLinks(";
        glue << "Particle, Tower)";
        glue << " VALUES (" << particle_foreignKey
             << ", " << SQLkey << ");";
        const string tmp2 = glue.str();
        sql = tmp2.c_str();
        /* Execute SQL statement */
        rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);

        check_execution(rc, zErrMsg);

        delete particle_print;
    }
}


void UpdateRecord(sqlite3 *db, string tableName, int recordKey, string fieldName, string newValue){
    char *zErrMsg = 0;
    int rc;
    const char *sql;

    /* Insert the particle, create a statment*/
    std::stringstream glue;
    glue << "UPDATE " << tableName;
    glue << " SET " << fieldName << " = " << newValue;
    glue << " WHERE ID = " << recordKey;
    const string tmp = glue.str();
    sql = tmp.c_str();

    /* Execute SQL statement */
    rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);

    check_execution(rc, zErrMsg);
}


void UpdateRecord(sqlite3 *db, string tableName, int recordKey, string fieldName, int newValue){
    std:stringstream inputValue;
    inputValue << newValue;
    UpdateRecord(db, tableName, recordKey, fieldName, inputValue.str());
}

void AnalyseEvents(ExRootTreeReader *treeReader, sqlite3 *db) {
    TClonesArray *branchEvent = treeReader->UseBranch("Event");
    TClonesArray *branchTrack = treeReader->UseBranch("Track");
    TClonesArray *branchTower = treeReader->UseBranch("Tower");
    TClonesArray *branchGenJet = treeReader->UseBranch("GenJet");
    TClonesArray *branchGenMissingET = treeReader->UseBranch("GenMissingET");
    TClonesArray *branchFatJet = treeReader->UseBranch("FatJet");
    TClonesArray *branchMissingET = treeReader->UseBranch("MissingET");
    TClonesArray *branchScalarHT = treeReader->UseBranch("ScalarHT");
    TClonesArray *branchParticle = treeReader->UseBranch("Particle");
    TClonesArray *branchElectron = treeReader->UseBranch("Electron");
    TClonesArray *branchPhoton = treeReader->UseBranch("Photon");
    TClonesArray *branchMuon = treeReader->UseBranch("Muon");
    TClonesArray *branchEFlowTrack = treeReader->UseBranch("EFlowTrack");
    TClonesArray *branchEFlowPhoton = treeReader->UseBranch("EFlowPhoton");
    TClonesArray *branchEFlowNeutralHadron = treeReader->UseBranch("EFlowNeutralHadron");
    TClonesArray *branchJet = treeReader->UseBranch("Jet");

    Long64_t allEntries = treeReader->GetEntries();

    cout << "** Chain contains " << allEntries << " events" << endl;

    GenParticle *particle;
    Electron *electron;
    Photon *photon;
    Muon *muon;

    Track *track;
    Tower *tower;

    Jet *jet;
    TObject *object;

    TLorentzVector momentum;

    Float_t Eem, Ehad;
    Bool_t skip;

    Long64_t entry;

    Int_t i, j, pdgCode;
    
    // needed to match mothers and daughters
    // string chosen because we can print particles in a way that is hopefully unique
    std::map<string, int> particle_to_SQLkey; 
    int SQLkey = 0;
    string *particle_print = new string();
    int m1, m2, d1, d2;

    // Loop over all events
    //for(entry = 0; entry < allEntries; ++entry) {
    for(entry = 0; entry < 1; ++entry) {
        // Load selected branches with data from specified event
        treeReader->ReadEntry(entry);

        cout << "--	New event -- " << endl;

        cout << "Adding particles \n";
        //Loop over particle in the event
        //for(i=0; i < 2; i++){
        for(i=0; i < branchParticle->GetEntriesFast(); i++){
            particle = (GenParticle *) branchParticle->At(i);
            AddParticle(particle, SQLkey, db);
            PrintParticle(particle, -1, particle_print);
            if(particle_to_SQLkey.find(*particle_print) != particle_to_SQLkey.end()){
                cout << "ERROR! particle "<< *particle_print << " found twice!\n";
            }
            particle_to_SQLkey[*particle_print] = SQLkey;
            SQLkey++;
        }
        cout << "Particles added. \n"
            << "Looping again to assign heratage\n";
        // Not working because https://cp3.irmp.ucl.ac.be/projects/delphes/ticket/1138
        // Basically there is a problem in someone elses code..... can't use M1, M2, D1, D2
        // Check HEPMC file, check makeclass outputs
        for(i=0; i < branchParticle->GetEntriesFast(); i++){
            particle = (GenParticle *) branchParticle->At(i);
            PrintParticle(particle, -1, particle_print);
            int thisKey = particle_to_SQLkey[*particle_print];
            m1 = particle->M1; 
            //if(m1 == i){
            //    cout << "Particle " << i << " appears to be it's own M1?!";
            //}
            if(m1 > -1){
                particle = (GenParticle *) branchParticle->At(m1);
                PrintParticle(particle, -1, particle_print);
                int foreignKey = particle_to_SQLkey[*particle_print];
                UpdateRecord(db, "GenParticles", thisKey, "M1", foreignKey);
                /*
                if(particle->D1 != i and particle->D2 != i){
                    cout << "\n Problem, particle " << *particle_print << " not acknowldging daughter!\n";
                }
                */
            }
            m2 = particle->M2; 
            //if(m2 == i){
            //    cout << "Particle " << i << " appears to be it's own M2?!";
            //}
            if(m2 > -1){
                particle = (GenParticle *) branchParticle->At(m2);
                PrintParticle(particle, -1, particle_print);
                int foreignKey = particle_to_SQLkey[*particle_print];
                UpdateRecord(db, "GenParticles", thisKey, "M2", foreignKey);
                /*
                if(particle->D1 != i and particle->D2 != i){
                    cout << "\n Problem, particle " << *particle_print << " not acknowldging daughter!\n";
                }
                */
            }
            d1 = particle->D1; 
            //if(d1 == i){
            //    cout << "Particle " << i << " appears to be it's own D1?!";
            //}
            if(d1 > -1){
                particle = (GenParticle *) branchParticle->At(d1);
                PrintParticle(particle, -1, particle_print);
                int foreignKey = particle_to_SQLkey[*particle_print];
                UpdateRecord(db, "GenParticles", thisKey, "D1", foreignKey);
                /*
                if(particle->M1 != i and particle->M2 != i){
                    cout << "\n Problem, particle " << *particle_print << " not acknowldging mother!\n";
                }
                */
            }
            d2 = particle->D2; 
            //if(d2 == i){
            //    cout << "Particle " << i << " appears to be it's own D2?!";
            //}
            if(d2 > -1){
                particle = (GenParticle *) branchParticle->At(d2);
                PrintParticle(particle, -1, particle_print);
                int foreignKey = particle_to_SQLkey[*particle_print];
                UpdateRecord(db, "GenParticles", thisKey, "D2", foreignKey);
            }

        }
        cout << "Heratage established\n";

        /*
        //Loop over all the particles to confirm the have their own numbers
        cout << "Particle loop\n";
        for(i=0; i< branchParticle->GetEntriesFast(); i++){
            particle = (GenParticle *)branchParticle->At(i);
            PrintParticle(particle, -1, particle_print);
            int foreignKey = particle_to_SQLkey[*particle_print];
            cout << foreignKey << ", ";
            cout.flush();
        }
        */
        //Loop over all the tracks and fill that table up
        cout << "Adding tracks\n";
        for(i=0; i< branchTrack->GetEntriesFast(); i++){
            track = (Track *)branchTrack->At(i);
            AddTrack(track, &particle_to_SQLkey, i, db);
        }


        //Loop over all the towers and fill that table up
        cout << "Adding towers\n";
        for(i=0; i< branchTower->GetEntriesFast(); i++){
            tower = (Tower *)branchTower->At(i);
            AddTower(tower, &particle_to_SQLkey, i, db);
        }

        // Loop over all jets in event
        /*
        for(i = 0; i < branchJet->GetEntriesFast(); ++i) {
            jet = (Jet*) branchJet->At(i);

            momentum.SetPxPyPzE(0.0, 0.0, 0.0, 0.0);

            cout<<"Looping over jet constituents. Jet pt: "<<jet->PT<<", eta: "<<jet->Eta<<", phi: "<<jet->Phi<<endl;

            // Loop over all jet's constituents
            for(j = 0; j < jet->Constituents.GetEntriesFast(); ++j)
            {
                object = jet->Constituents.At(j);

                // Check if the constituent is accessible
                if(object == 0) continue;

                if(object->IsA() == GenParticle::Class())
                {
                    particle = (GenParticle*) object;
                    cout << "		GenPart pt: " << particle->PT << ", eta: " << particle->Eta << ", phi: " << particle->Phi << endl;
                    momentum = particle->P4();
                }
                else if(object->IsA() == Track::Class())
                {
                    track = (Track*) object;
                    //cout << "Track~~~~~~" << endl;
                    momentum += track->P4();
                    //PrintTrack(track, 0);
                }
                else if(object->IsA() == Tower::Class())
                {
                    tower = (Tower*) object;
                    //cout << "Tower~~~~" << endl;
                    momentum = tower->P4();
                    //PrintTower(tower, 0);
                }
                else
                {
                    cout << "WTF is this?" << endl;
                }
            }
        }
        */
    }
    delete particle_print;
}

//------------------------------------------------------------------------------

void MakeCheckFile(ExRootTreeReader *treeReader, const char* rootFile) {
    //make an output file to write to
    string outName(rootFile);
    ConvertFilename(outName, "_cpp.txt");
	FindAndReplaceAll(outName, "~/", "/home/henry/");
    
    ofstream outFile;
    outFile.open(outName);

    TClonesArray *branchEvent = treeReader->UseBranch("Event");
    TClonesArray *branchTrack = treeReader->UseBranch("Track");
    TClonesArray *branchTower = treeReader->UseBranch("Tower");
    TClonesArray *branchGenJet = treeReader->UseBranch("GenJet");
    TClonesArray *branchGenMissingET = treeReader->UseBranch("GenMissingET");
    TClonesArray *branchFatJet = treeReader->UseBranch("FatJet");
    TClonesArray *branchMissingET = treeReader->UseBranch("MissingET");
    TClonesArray *branchScalarHT = treeReader->UseBranch("ScalarHT");
    TClonesArray *branchParticle = treeReader->UseBranch("Particle");
    TClonesArray *branchElectron = treeReader->UseBranch("Electron");
    TClonesArray *branchPhoton = treeReader->UseBranch("Photon");
    TClonesArray *branchMuon = treeReader->UseBranch("Muon");
    TClonesArray *branchEFlowTrack = treeReader->UseBranch("EFlowTrack");
    TClonesArray *branchEFlowPhoton = treeReader->UseBranch("EFlowPhoton");
    TClonesArray *branchEFlowNeutralHadron = treeReader->UseBranch("EFlowNeutralHadron");
    TClonesArray *branchJet = treeReader->UseBranch("Jet");

    Long64_t allEntries = treeReader->GetEntries();

    cout << "** Chain contains " << allEntries << " events" << endl;

    GenParticle *particle;
    Long64_t entry;

    Int_t i, j, pdgCode;

    // Loop over all events
    //for(entry = 0; entry < allEntries; ++entry) {
    for(entry = 0; entry < 1; ++entry) {
        // Load selected branches with data from specified event
        treeReader->ReadEntry(entry);

        cout << "--	New event -- " << endl;

        cout << "Printing Particles \n";
        //Loop over particle in the event
        //for(i=0; i < 2; i++){
        for(i=0; i < branchParticle->GetEntriesFast(); i++){
            particle = (GenParticle *) branchParticle->At(i);
            string outString;
            string *particle_print = new string();
            PrintParticle(particle, -1, particle_print) ;
            outFile << *particle_print << endl;
            delete particle_print;
        }
        cout << "Particles printed. \n";
    }
    outFile.close();
}


void PlayWithTracks(ExRootTreeReader *treeReader, const char* rootFile) {
    TClonesArray *branchEvent = treeReader->UseBranch("Event");
    TClonesArray *branchTrack = treeReader->UseBranch("Track");
    TClonesArray *branchTower = treeReader->UseBranch("Tower");
    TClonesArray *branchGenJet = treeReader->UseBranch("GenJet");
    TClonesArray *branchGenMissingET = treeReader->UseBranch("GenMissingET");
    TClonesArray *branchFatJet = treeReader->UseBranch("FatJet");
    TClonesArray *branchMissingET = treeReader->UseBranch("MissingET");
    TClonesArray *branchScalarHT = treeReader->UseBranch("ScalarHT");
    TClonesArray *branchParticle = treeReader->UseBranch("Particle");
    TClonesArray *branchElectron = treeReader->UseBranch("Electron");
    TClonesArray *branchPhoton = treeReader->UseBranch("Photon");
    TClonesArray *branchMuon = treeReader->UseBranch("Muon");
    TClonesArray *branchEFlowTrack = treeReader->UseBranch("EFlowTrack");
    TClonesArray *branchEFlowPhoton = treeReader->UseBranch("EFlowPhoton");
    TClonesArray *branchEFlowNeutralHadron = treeReader->UseBranch("EFlowNeutralHadron");
    TClonesArray *branchJet = treeReader->UseBranch("Jet");

    Long64_t allEntries = treeReader->GetEntries();

    cout << "** Chain contains " << allEntries << " events" << endl;

    GenParticle *particle;
    Track *track;
    Long64_t entry;

    Int_t i, j, pdgCode;

    // Loop over all events
    //for(entry = 0; entry < allEntries; ++entry) {
    for(entry = 0; entry < 1; ++entry) {
        // Load selected branches with data from specified event
        treeReader->ReadEntry(entry);

        cout << "--	New event -- " << endl;

        cout << "Printing tracks \n";
        //Loop over particle in the event
        //for(i=0; i < 2; i++){
        for(i=0; i < branchTrack->GetEntriesFast(); i++){
            track = (Track *) branchTrack->At(i);
            particle = (GenParticle *)track->Particle.GetObject();
            cout << "Track ";
            PrintTrack(track, 0);
            PrintParticle(particle, 0);
        }
        cout << "Tracks printed. \n";
    }
}



void to_sql()
{
    const char *inputFile = "~/lazy/29delphes_events.root";
    //gSystem->Load("/usr/lib/x86_64-linux-gnu/libsqlite3.so"); // This dosn't work here, have to do it in cint
    gSystem->Load("libDelphes");

    sqlite3 *db = CreateDatabase(inputFile);


    TChain *chain = new TChain("Delphes");
    chain->Add(inputFile);

    ExRootTreeReader *treeReader = new ExRootTreeReader(chain);

    AnalyseEvents(treeReader, db);
    //MakeCheckFile(treeReader, inputFile);
    //PlayWithTracks(treeReader, inputFile);

    cout << "** Exiting..." << endl;


    delete treeReader;
    delete chain;
}

//------------------------------------------------------------------------------
