
import argparse
import pandas as pd
import itertools

# Function that determines if a functional dependency is non trivial
# A functional dependency is non trivial if for X -> Y, Y is not a subset of X
def non_triv(f):

    for rhs in f['rhs']:
        if rhs in f['lhs']:
            #print(f"Non triv returns false for {f}")
            return False
    
    #print(f"Non triv returns true for {f}")
    return True

# Function that takes in the functional dependency dict object and print it out in the "X -> Y" format
def print_func_dep(f):
    str = ''
    for i, lhs in enumerate(f['lhs']):
        str = str + lhs
        if i != len(f['lhs'])-1:
            str = str + ', '
    
    if f['multi']:
        str = str + " ->> "
    else:
        str = str + " -> "
    
    for i, rhs in enumerate(f['rhs']):
        str = str + rhs
        if i != len(f['rhs'])-1:
            str = str + ', '
    
    print(str)

# Function that takes as input what normal form the user wants to normalize up to, and then runs all the necessary normalization functions on the inputted relation
def normalize_from_input(rel, form):

    if form == '5':
        rel.normalize_1nf()
        rel.normalize_2nf()
        rel.normalize_3nf()
        rel.normalize_bcnf()
        rel.normalize_4nf()
        rel.normalize_5nf()

    elif form == '4':

        rel.normalize_1nf()
        rel.normalize_2nf()
        rel.normalize_3nf()
        rel.normalize_bcnf()
        rel.normalize_4nf()
    
    elif form == 'b' or form == 'B':

        rel.normalize_1nf()
        rel.normalize_2nf()
        rel.normalize_3nf()
        rel.normalize_bcnf()
    
    elif form == '3':

        rel.normalize_1nf()
        rel.normalize_2nf()
        rel.normalize_3nf()
    
    elif form == '2':

        rel.normalize_1nf()
        rel.normalize_2nf()
    
    elif form == '1':

        rel.normalize_1nf()

# Helper function for normalizing tables with multiple attributes
# Any row with a multiple attribute thats detected, has the multiple attribute broken up across different rows with the same values for other columns
def separate_multi(table):

    # Iterate through all rows in the tables
    for idx, row in table.iterrows():

        for idx_, col in enumerate(row):
            
            # A column with multiple attributes has been detected
            if "|" in str(col):
                #print(row[idx_])
                
                # split up multivalued attribute
                vals = str(col).split("|")

                for i, v in enumerate(vals):
                    # copy exisiting row
                    tmprow = row

                    # replace value at the column index of the multivalue attribute with just one of the attributes
                    tmprow[idx_] = v

                    # add new row at the end of table with newly generated row
                    table.loc[len(table)] = tmprow
            
                # delete original row with mutlival
                table = table.drop(idx)
    
    return table

# Helper function to iterate through a list 2 elements at a time
# Used in 4NF normalizer to normalize each 2 mvds together
def chunker(list, size):
    return(list[pos:pos + size] for pos in range(0, len(list), size))

# Generate all possible projections of a relation given a list of columns
# Used in 5NF normalizer
def permutations(cols):

    # Helper function to convert the list of tuples from the combination permutation results into a list of lists
    def tup_to_list(tuples):
        l = []
        for tup in tuples: l.append(list(tup))
        return l
    
    perms = []

    # If a table only has 2 columns, there are only two possible projections
    if len(cols) == 2:
        perms.append(([cols[0]], [cols[1]]))
        perms.append(([cols[1]], [cols[0]]))
    
    else:

        # Split cols each into their own table
        fullsplit = []
        for c in cols:
            fullsplit.append([c])
        
        perms.append(fullsplit)

        # Generate all permuations of the columns with one column left out
        for i, c in enumerate(cols):
            #sub = list(set(cols) - set(c))
            sub = [item for item in cols if item != c]
            perms.append((sub, [c]))
    
        # Generate all unique combinations of cols
        for i in range(2, len(cols)):
            
            # generate all combinitons of a particluar size i
            comb = list(itertools.combinations(cols, i))

            # generate all permuations of the combinations of size i
            combperms = list(itertools.permutations(comb, i))

            for cp in combperms:

                # Permutation only gets added to the list if its unique (not already in list)
                add = True

                # Generate all permuations of this specific permuation to see if any are already stored
                cpperms = list(itertools.permutations(cp, len(cp)))
                for cpp in cpperms:
                    if list(cpp) in perms:
                        #print(f'{cpp} already in perms list')
                        add = False

                # If no permuations of this combination have been stored yet, store it
                if add: perms.append(list(tup_to_list(cp)))

    return perms

# Function to natural join tables together based on common column
# Joins two tables at a time and recursively joins any amount of tables given
def join(tables):

    # Helper function to naturally join together two tables at once
    def natural_join(df1, df2):
        # Find common columns for the natural join
        common_columns = list(set(df1.columns) & set(df2.columns))

        joined_df = pd.DataFrame()
        if len(common_columns) > 0:
        
            # Perform the natural join based on common columns
            joined_df = pd.merge(df1, df2, on=common_columns)
        
        # An empty dataframe is returned if there are no common cols
        return joined_df

    # Base case: if there's only one dataframe, return it
    if len(tables) == 1:
        return tables[0]
    
    # recursively perform natural join operation
    joined_df = natural_join(tables[0], tables[1])

    if len(joined_df.index):
    
        # Join with remaining dataframes
        for df in tables[2:]:
            joined_df = natural_join(joined_df, df)
    
    return joined_df

# The relation table class represents a singular table and also stores the values for the table primary key, functional dependencies, foreign keys, and table identifier
# The relation table class also stores all the functions to check the table for a particular normal form
class Relation_Table:
    def __init__(self, table, func_deps, key, name = 0, fk = []):
        self.table = table
        self.func_deps = func_deps
        self.key = key
        self.name = f"Table{name}"
        self.fk = fk

    # EXTRA CREDIT FUNCTION
    # Detects mutlivalue dependencies within table based on values
    def mvd_detector(self):

        mvd = []
        # get column names for table
        columns = list(self.table.columns)

        # iterate across three columns at at time
        for cola in columns:
            for colb in columns:
                for colc in columns:
                    
                    # iterate across 4 columns at a time while iterating across columns
                    for idx1, t1 in self.table.iterrows():
                        for idx2, t2 in self.table.iterrows():
                            for idx3, t3, in self.table.iterrows():
                                for idx4, t4 in self.table.iterrows():
                                    
                                    # first condition for mvd -> t1[a] = t2[a] = t3[a] = t4[a]
                                    if t1[cola] == t2[cola] and t1[cola] == t3[cola] == t4[cola]:

                                        # second condition for mvd -> t1[b] = t3[b] AND t2[b] = t4[b]
                                        if t1[colb] == t3[colb] and t2[colb] == t4[colb]:

                                            # third condition for mvd -> t2[c] = t3[c] AND t1[c] = t4[c]
                                            if t2[colc] == t3[colc] and t1[colc] == t4[colc]:

                                                #print("MULTIVALUE DEPENDENCY FOUND!")
                                                #print(f"ColA: {cola}\nColB: {colb}\nColC: {colc}")

                                                if (cola, colb, colc) not in mvd:
                                                    mvd.append((cola, colb, colc))
        
        print(f"{len(mvd)} MVDs found")

    def one_nf_check(self):
        # store a list of which columns contain multivalued attributes
        multivalue_attributes = []

        # for each row in the table, iterate through each column looking for multivalued dependencies
        for index, row in self.table.iterrows():
            for idx, col in enumerate(row):

                # a column with multiple values has been found
                if len(str(col).split('|')) > 1:
                    
                    multivalue_attributes.append(list(self.table.columns)[idx])
        
        # if there are more than no multivalue_attributes, function returns false
        if len(multivalue_attributes)>0:
            return (False, multivalue_attributes)
        
        # if there are no multivalue attributes, the table is in 1nf
        else:
            return (True, multivalue_attributes)
        
    def two_nf_check(self):
        # list containing all partial functional dependencies
        partials = []

        for f in self.func_deps:
            for lhs in f['lhs']:

                # One attribute from the lhs is a part of the primary key
                if lhs in key:
                    # One attribute from the lhs is in the primary key but another attribute isn't, thus creating a partial functional dependency
                    if lhs != key:
                                            
                        partials.append(f)
        
        if len(partials)>0:
            return (False, partials)
        else:
            return (True, partials)
    
    def three_nf_check(self):
        trans = []

        # For every non-trivial functional dependency X -> Y, either X must be a superkey or Y is a prime attribute
        for f in self.func_deps:
            # Check that f is non-trivial
            if not f['multi']:
                #print(f'{f} passed multi check')
                if non_triv(f):
                    
                    # Check if X is a super key
                    if f['lhs'] != self.key:

                        # Check if Y is a prime attribute (each element of Y is part of some candidate key)
                        for rhs in f['rhs']:                            

                            # This element of Y is not in the key, therefore this function breaks 3nf
                            if not rhs in self.key:
                                trans.append(f)
                    
                   
            
        if len(trans)>0:
            return (False, trans)
        else:
            return (True, trans)
    
    def bcnf_nf_check(self):

        # list of attributes that are not dependent on the primary key
        bcnf = []

        # for any f X->Y, X must be in the key
        for f in self.func_deps:
            
            if not f['multi']:
                if f['lhs'] != self.key:
                    bcnf.append(f)
        
        if len(bcnf)>0:
            return (False, bcnf)
        else:
            return (True, bcnf)

    def four_nf_check(self):
        multis = []

        # check to see if any of the functional dependencies for a table have the MVD signifier ->>
        for f in self.func_deps:
            if f['multi']:
                multis.append(f)

        if len(multis)>0:
            return (False, multis)
        else:
            return (True, multis)
    
    def five_nf_check(self):

        columns = list(self.table.columns)
        colperms = permutations(columns)

        for c in colperms:            
            # Generate tables from column permutations
            proj = []
            for colset in c:

                #print(f"Making table out of rows: {colset}")
                new_tbl = pd.DataFrame()

                # For each column in colset, copy column from starting table
                for col in colset:
                    new_tbl[col] = self.table[col]
               
                proj.append(new_tbl)
            
            # Join the tables back together using nat join
            joined = join(proj)

            # If joined table is equal to the original table, this is a valid join depenency
            if len(joined.index):

                if joined.equals(self.table):
                    print("PROPER VALID JOIN DEP FOUND: ")
                    print(c)
                    return c        

        # None of the column permutations returned a valid join dependency
        return None

# The relation class simply stores a list of all the tables in a given relation, and stores the functions to normalize for particular forms
class Relation:
    def __init__(self, table_list):
        self.table_list = table_list
    
    # Function to print out each table in the relation and all related information
    def print_rel(self):

        print(f"\n--Relation has {len(self.table_list)} table(s)--")

        for idx, tbl in enumerate(self.table_list):
            print(f"\n-=-=-=-=- {tbl.name} -=-=-=-=- ")
            print(tbl.table)
            print(f"\nPrimary Key: {tbl.key}")
            print(f"\nFunctional Dependencies: ")
            for f in tbl.func_deps:
                print_func_dep(f)
            print(f"\nForeign Keys: ")
            for fk in tbl.fk:
                print(f"FOREIGN KEY ({fk['Column']}) REFERENCES {fk['RefTable']}({fk['RefColumn']})")
            
            print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        
        print()
    
    # Check and normalize each table in list to 1NF
    def normalize_1nf(self):
        
        # check each table in the relation
        for index, tbl in enumerate(self.table_list):
            valid, multivals = tbl.one_nf_check()
            
            # the table is not in 1nf -> normalize it
            if not valid:
                tmp_table = tbl.table    

                # Separate the multivalue attributes out of tables
                new_tbl = separate_multi(tmp_table)

                # replace table in list with updated table
                self.table_list[index].table = new_tbl
    
    # Check and normalize each table in list to 2NF
    def normalize_2nf(self):
        new_tbl_list = []
        
        # check each table in the relation
        for index, tbl in enumerate(self.table_list):
            valid, partials = tbl.two_nf_check()
            
            # table is not in 2nf -> normalize it
            if not valid:
                tmp_table = tbl.table
                tmp_funcdep = tbl.func_deps
                tbl_num = 0
                print(f"Base table will be num: {tbl_num}")

                # create new table for each partial dependency
                for p in partials:
                    if not p['multi']:
                        #print(f"Table has partial dep: {p}")
                    
                        cols = []
                        pkey = []
                        funcs = [p]
                        colnames = []

                        #print(f"Normalizing partial functional dependency: {p}")
                        #print(tmp_table)
                        # Create new table from removed column and primary key
                        new_tbl = pd.DataFrame()
                        # Generate foreign key relations for new table
                        newfk = []
                        
                        # generate keys from lhs of dependency
                        for lhs in p['lhs']:
                            #print(f"p[lhs] == {lhs}")
                            cols.append(tmp_table[lhs])
                            

                            if lhs in tbl.key:
                                newfk.append({'Column': lhs,
                                        'RefTable': f"Table{tbl_num}",
                                        'RefColumn': lhs})
                            else:
                                colnames.append(lhs)

                            pkey.append(lhs)
                        
                        # add dependent cols and then delete them from origin table
                        for rhs in p['rhs']:
                            cols.append(tmp_table[rhs])
                            del tmp_table[rhs]

                        # add saved columns to new table 
                        for c in cols:
                            new_tbl.insert(len(list(new_tbl.columns)), c.name, c, True)

                        # find the functional dependency being normalized and remove it from tmp list
                        #print(f"Colnames: {colnames}")
                        for i, f in enumerate(tmp_funcdep):
                            #print(f"Functional Dependency in tmp_funcdep: {print_func_dep(f)}")
                            if f == p:
                                tmp_funcdep.pop(i)
                            
                            # move other functional dependencies relating to attribute being moved to new table
                            if f['rhs'] in colnames:
                                #print("MOVING FUNCTIONAL DEP")
                                func = tmp_funcdep.pop(i)
                                funcs.append(func)
                            
                            for rhs in f['rhs']:
                                if rhs in colnames:
                                    if not f['multi']:
                                        funcs.append(tmp_funcdep.pop(i))
                                        break
                        
                        # decide if mvds need to be removed from tmp_table
                        for i, f in enumerate(tmp_funcdep):
                            if f['multi']:

                                # if the lhs has been removed, delete mvd
                                for lhs in f['lhs']:
                                    if lhs not in list(tmp_table.columns):

                                        for i_, f_, in enumerate(tmp_funcdep):
                                            if f_['multi'] and f_['lhs'] == f['lhs']:

                                                tmp_funcdep.pop(i)
                                                tmp_funcdep.pop(i_)
                                
                                # if the lhs has been removed, delete mvd
                                for lhs in f['rhs']:
                                    if lhs not in list(tmp_table.columns):

                                        for i_, f_, in enumerate(tmp_funcdep):
                                            if f_['multi'] and f_['lhs'] == f['lhs']:

                                                tmp_funcdep.pop(i)
                                                tmp_funcdep.pop(i_)
                        
                        #print("New table:")
                        #print(new_tbl)

                        # TODO: Fix table numbering system
                        new_tbl_list.append(Relation_Table(new_tbl, funcs, pkey, name=len(new_tbl_list)+1, fk=newfk))
                
                new_tbl_list.append(Relation_Table(tmp_table, tmp_funcdep, tbl.key, name=0))
            
            else:
                new_tbl_list.append(tbl)

        self.table_list = new_tbl_list

    # Check and normalize each table in list to 3NF
    def normalize_3nf(self):
        new_tbl_list = []
        
        # check each table in the relation
        for index, tbl in enumerate(self.table_list):
            valid, trans = tbl.three_nf_check()
            
            # table is not in 2nf -> normalize it
            if not valid:
                tmp_table = tbl.table
                tmp_funcdep = tbl.func_deps
                tbl_num = 0
                print(f"Base table will be num: {tbl_num}")

                # create new table for each partial dependency
                for t in trans:
                    if not t['multi']:
                        print(f"Table has transitive dep: {t}")
                    
                        cols = []
                        pkey = []
                        funcs = [t]
                        colnames = []

                        #print(f"Normalizing partial functional dependency: {p}")
                        #print(tmp_table)
                        # Create new table from removed column and primary key
                        new_tbl = pd.DataFrame()
                        # Generate foreign key relations for new table
                        newfk = []
                        
                        # generate keys from lhs of dependency
                        for lhs in t['lhs']:
                            #print(f"p[lhs] == {lhs}")
                            cols.append(tmp_table[lhs])
                            

                            if lhs in tbl.key:
                                newfk.append({'Column': lhs,
                                        'RefTable': f"Table{tbl_num}",
                                        'RefColumn': lhs})
                            else:
                                colnames.append(lhs)

                            pkey.append(lhs)
                        
                        # add dependent cols and then delete them from origin table
                        for rhs in t['rhs']:
                            cols.append(tmp_table[rhs])
                            del tmp_table[rhs]

                        # add saved columns to new table 
                        for c in cols:
                            new_tbl.insert(len(list(new_tbl.columns)), c.name, c, True)

                        # find the functional dependency being normalized and remove it from tmp list
                        #print(f"Colnames: {colnames}")
                        for i, f in enumerate(tmp_funcdep):
                            #print(f"Functional Dependency in tmp_funcdep: {print_func_dep(f)}")
                            if f == t:
                                tmp_funcdep.pop(i)
                            
                            # move other functional dependencies relating to attribute being moved to new table
                            if f['rhs'] in colnames:
                                #print("MOVING FUNCTIONAL DEP")
                                func = tmp_funcdep.pop(i)
                                funcs.append(func)
                            
                            for rhs in f['rhs']:
                                if rhs in colnames:
                                    if not f['multi']:
                                        funcs.append(tmp_funcdep.pop(i))
                                        break
                        
                        # decide if mvds need to be removed from tmp_table
                        for i, f in enumerate(tmp_funcdep):
                            if f['multi']:

                                # if the lhs has been removed, delete mvd
                                for lhs in f['lhs']:
                                    if lhs not in list(tmp_table.columns):

                                        for i_, f_, in enumerate(tmp_funcdep):
                                            if f_['multi'] and f_['lhs'] == f['lhs']:

                                                tmp_funcdep.pop(i)
                                                tmp_funcdep.pop(i_)
                                
                                # if the lhs has been removed, delete mvd
                                for lhs in f['rhs']:
                                    if lhs not in list(tmp_table.columns):

                                        for i_, f_, in enumerate(tmp_funcdep):
                                            if f_['multi'] and f_['lhs'] == f['lhs']:

                                                tmp_funcdep.pop(i)
                                                tmp_funcdep.pop(i_)
                        
                        #print("New table:")
                        #print(new_tbl)

                        # TODO: Fix table numbering system
                        new_tbl_list.append(Relation_Table(new_tbl, funcs, pkey, name=len(new_tbl_list)+1, fk=newfk))
                
                new_tbl_list.append(Relation_Table(tmp_table, tmp_funcdep, tbl.key, name=tbl_num))
            
            else:
                new_tbl_list.append(tbl)

        self.table_list = new_tbl_list

    # Check and normalize each table in list to BCNF
    def normalize_bcnf(self):
        new_tbl_list = []
        
        # check each table in the relation
        for index, tbl in enumerate(self.table_list):
            valid, bcnf = tbl.bcnf_nf_check()
            
            # table is not in 2nf -> normalize it
            if not valid:
                tmp_table = tbl.table
                tmp_funcdep = tbl.func_deps
                tbl_num = len(self.table_list)
                #print(f"Base table will be num: {tbl_num}")

                print(tbl.table)
                print(f"This table is experiencing BCNF discrepensy")

                # create new table for each partial dependency
                for b in bcnf:
                    print(f"{b} breaks bcnf rules")
                    cols = []
                    pkey = []

                    # Create new table from removed column and primary key
                    new_tbl = pd.DataFrame()

                    # Generate foreign key relations for new table
                    newfk = []
                    
                    # generate keys from lhs of dependency
                    for lhs in b['lhs']:
                        cols.append(tmp_table[lhs])

                        # Write foreign key from table keys
                        if lhs in tbl.key:
                            newfk.append({'Column': lhs,
                                    'RefTable': f"Table{tbl_num}",
                                    'RefColumn': lhs})
                        pkey.append(lhs)
                    
                    # add dependent cols and then delete them from origin table
                    for rhs in b['rhs']:
                        cols.append(tmp_table[rhs])
                        del tmp_table[rhs]

                    # add saved columns to new table 
                    for c in cols:
                        new_tbl.insert(len(list(new_tbl.columns)), c.name, c, True)

                    # find the functional dependency being normalized and remove it from tmp list
                    for i, f in enumerate(tmp_funcdep):
                        if f == b:
                            tmp_funcdep.pop(i)

                    # TODO: Fix table numbering system
                    new_tbl_list.append(Relation_Table(new_tbl, [b], pkey, name=len(new_tbl_list)+1, fk=newfk))
            
                new_tbl_list.append(Relation_Table(tmp_table, tmp_funcdep, tbl.key, name=tbl_num))
        
            else:
                new_tbl_list.append(tbl)

        self.table_list = new_tbl_list

    # Check and normalize each table in list to 4NF
    def normalize_4nf(self):
        new_tbl_list = []
        
        # check each table in the relation
        for index, tbl in enumerate(self.table_list):
            valid, multis = tbl.four_nf_check()
            
            # table is not in 4nf -> normalize it
            if not valid:

                # Copy current table object for editing
                tmp_table = tbl.table
                tmp_funcdep = tbl.func_deps
                tbl_num = 0

                # 4nf normalizer works in chunks of two
                chunks = chunker(multis, 2)
                for m1, m2 in chunks:

                    # Set the multivalue dependency indicator for m1 and m2 to false since they no longer will be
                    m2['multi'] = False
                    m1['multi'] = False
                    cols = []
                    pkey = []
                    funcs = [m2]
                    colnames = []

                    # Move m2 to its own table
                    new_tbl = pd.DataFrame()
                    # Generate foreign key relations for new table
                    newfk = []
                    
                    # generate keys from lhs of dependency
                    for lhs in m2['lhs']:
                        #print(f"p[lhs] == {lhs}")
                        cols.append(tmp_table[lhs])
                        

                        # Write foreign key frome table key
                        if lhs in tbl.key:
                            newfk.append({'Column': lhs,
                                    'RefTable': f"Table{tbl_num}",
                                    'RefColumn': lhs})
                        else:
                            colnames.append(lhs)

                        # Set primary key to lhs of dependency
                        pkey.append(lhs)
                    
                    # add dependent cols and then delete them from origin table
                    for rhs in m2['rhs']:
                        cols.append(tmp_table[rhs])
                        del tmp_table[rhs]

                    # add saved columns to new table 
                    for c in cols:
                        new_tbl.insert(len(list(new_tbl.columns)), c.name, c, True)

                    # find the functional dependency being normalized and remove it from tmp list
                    #print(f"Colnames: {colnames}")
                    for i, f in enumerate(tmp_funcdep):
                        #print(f"Functional Dependency in tmp_funcdep: {print_func_dep(f)}")
                        if f == m2:
                            tmp_funcdep.pop(i)
                        
                        # move other functional dependencies relating to attribute being moved to new table
                        if f['rhs'] in colnames:
                            #print("MOVING FUNCTIONAL DEP")
                            func = tmp_funcdep.pop(i)
                            funcs.append(func)
                        
                        for rhs in f['rhs']:
                            if rhs in colnames:
                                if not f['multi']:
                                    funcs.append(tmp_funcdep.pop(i))
                                    break
                    
                    # Add newly generate table to table list
                    new_tbl_list.append(Relation_Table(new_tbl, funcs, pkey, name=len(new_tbl_list)+1, fk=newfk))

                # Add table that columns were removed from to table list
                new_tbl_list.append(Relation_Table(tmp_table, tmp_funcdep, tbl.key, name=tbl_num))
            
            else:
                new_tbl_list.append(tbl)

        self.table_list = new_tbl_list

    # Check and normalize each table in list to 5NF
    def normalize_5nf(self):
        new_tbl_list = []
        
        # check each table in the relation
        for index, tbl in enumerate(self.table_list):
            vjd = tbl.five_nf_check()
            
            # a valid join dep has been found for table
            if vjd:

                # Make a new table for each col perm in the join dep
                for perm in vjd:
                    funcs = []
                    pkey = self.key
                    newfk = []

                    new_tbl = pd.DataFrame()

                    # Create new table from cols in original table for each col in the permutation
                    for col in perm:
                        new_tbl[col] = self.table[col]
                    
                    # Add newly generate table to table list
                    new_tbl_list.append(Relation_Table(new_tbl, funcs, pkey, name=len(new_tbl_list)+1, fk=newfk))

            else:
                new_tbl_list.append(tbl)

    # Function to generate CREATE TABLE statements for each table in the relations table list
    # SQL statements are sent to common output and written to the file sqloutfile.txt
    def generate_sql(self):

        # This just opens the file in write mode and then closes it to blank it out before writing these tables
        with open("sqloutfile.txt", 'w') as f:
            pass
        
        
        for idx, tbl in enumerate(self.table_list):
            lines = []
            cols = list(tbl.table.columns)

            #print("GENERATING SQL FOR TABLE: ")
            #print(tbl.table)

            #print(f"CREATE TABLE Table#{tbl.name} (")
            lines.append(f"CREATE TABLE {tbl.name} (")
            
            # Table only has one primary key
            if len(tbl.key)==1:
                for c in cols:  

                    # write primary key
                    if c in tbl.key:
                        lines.append(f"\t{c} VARCHAR(255) PRIMARY KEY")
                        
                    else:
                        lines.append(f"\t{c} VARCHAR(255) NOT NULL")
            
            # Table has multiple primary key -> Denote them at end of the definition as tuple
            else:
                for c in cols:
                    lines.append(f"\t{c} VARCHAR(255) NOT NULL")
                
                # Output primary key tuple
                keystr = f"\tPRIMARY KEY ("
                for i, k in enumerate(tbl.key):
                    keystr = keystr + k
                    if i != len(tbl.key)-1:
                        keystr = keystr + ", "
                    else:
                        keystr = keystr + ")"
                        lines.append(keystr)
            
            # Output foreign keys
            for f in tbl.fk:
                lines.append(f"\tFOREIGN KEY ({f['Column']}) REFERENCES {f['RefTable']}({f['RefColumn']})")

            lines.append(");")

            # Lines are stored in an array first and then printed so that the program can detect when its on the last line in a 
            # tables definition so it knows to not write a comma at the end of that line
            with open("sqloutfile.txt", 'a') as f:
                for lnum, l in enumerate(lines):

                    if lnum != len(lines) - 2 and lnum != len(lines) - 1 and lnum != 0:

                        print(l)
                        f.write(l + ",\n")
                    
                    else:

                        print(l)
                        f.write(l + "\n")
                
                print()
                f.write('\n')

                
# Initialize input arguments for user
parser=argparse.ArgumentParser()
parser.add_argument("--tablefile", help="Name of file in folder to use as input for table")
parser.add_argument("--form", help="Integer form of highest normal form to achieve")
parser.add_argument("--key", help="Primary Key of table (can be compound)")
parser.add_argument("--inputfile", help="Name of file that stores functional dependencies")
parser.add_argument("--check", help="Check what normal form the inputted table is in")
args=vars(parser.parse_args())

# Parse input table 
if args['tablefile']:
    # get file name from user input
    file_name = args['tablefile']
    try:
        file = open(file_name)
    except:
        print('Error opening file selected, aborting...')
        exit()

# If no input table is specified, the program will use the file named 'exampleInputTable.csv' if it exists
else:
    try:
        file_name = "exampleInputTable.csv"
        file = open(file_name)
    except:
        print('Error opening default file, aborting...')
        exit()

# Read in table from specified input file
table = pd.read_csv(open(file_name, 'r'), na_filter=False)

# Generate list of column names from inputted table
columns = list(table.columns)

# Parse which highest form the user would like to normalize to
if args['form']:
    try:
        form = int(args['form'])
        form = args['form']
    except:

        if args['form'] == 'b' or args['form'] == 'B':
            form = 'b'
        else:
            print('Error with normal form input, aborting...')
            exit()
else:
    # default to using 4th normal form if none other is given
    form = '4'

# Parse primary key input from user
if args['key']:

    # if theres a comma in the key input, its multivalued
    if "," in args['key']:
        key = args['key'].split(',')

        # If a specified key is not in the table columns, the program will exit
        for k in key:
            if k not in columns:
                print(f"Column {k} not found in table, aborting...")
                exit()
    
    else:
        key = args['key']
        if key not in columns:
                print(f"Column {key} not found in table, aborting...")
                exit()

else:
    print("No key submitted, aborting...")
    exit()

# Parse file inputted containing functional dependencies
if args['inputfile']:

    try:
        inputfile = open(args['inputfile'])
    
    except:
        print("Error opening input file, aborting...")
        exit()

else:
    print("No input file submitted, aborting...")

# Parse functional dependencies
func_deps = []
for line in inputfile.readlines():

    # Read in functional dependency from line in file and write to to a dict
    if '-> ' in line:
        lhs = line.split('->')[0]
        rhs = line.split('->')[1]
        #print(f'lhs: {lhs} | rhs: {rhs}')

        func = {
            'lhs': lhs.replace(' ', '').replace('\n', '').split(','),
            'rhs': rhs.replace(' ', '').replace('\n', '').split(','),
            'multi': False
        }

        func_deps.append(func)
    
    elif '->>' in line:
        lhs = line.split('->>')[0]
        rhs = line.split('->>')[1]
        #print(f'lhs: {lhs} | rhs: {rhs}')

        func = {
            'lhs': lhs.replace(' ', '').replace('\n', '').split(','),
            'rhs': rhs.replace(' ', '').replace('\n', '').split(','),
            'multi': True
        }

        func_deps.append(func)

# create initial table and relation objects from user input
table = pd.read_csv(open(file_name, 'r'), na_filter=False)
rel_tbl = Relation_Table(table=table, func_deps=func_deps, key=key)
relation = Relation(table_list=[rel_tbl])

# Print the inputted table before any normalization steps
print(f"Relation before normalization: ")
relation.print_rel()

# If the user has submitted that they would like to check the highest form of input table, run every check function
if args['check'] == 'True':

    onenf, _ = rel_tbl.one_nf_check()

    if onenf:
        twonf, _ = rel_tbl.two_nf_check()

        if twonf:
            threenf, _ = rel_tbl.three_nf_check()

            if threenf:
                bcnf, _ = rel_tbl.bcnf_nf_check()

                if bcnf:
                    fournf, _ = rel_tbl.four_nf_check()

                    if fournf:
                        fivenf, _ = rel_tbl.five_nf_check()

                        if fivenf:
                            print("Table is in 5nf - Highest achievable form in program!")
                        
                        else:
                            print("Table's highest form is 4nf")
                    
                    else:
                        print("Table's highest form is bcnf")
                
                else:
                    print("Table's highest form is 3nf")

            else:
                print("Table's highest form is 2nf")
        
        else:
            print("Table's highest form is 1nf")

    else:
        print("Table is not in 1NF")

else:
    # run function to normalize table up to form selected by user
    print(f"User has selected to normalize up to {form}\n")
    normalize_from_input(relation, form)

    # show final table
    print(f"Final normalized table: ")
    relation.print_rel()

    # show generated sql code
    print(f"Generate SQL code for final relation format:")
    relation.generate_sql()