I found out how you have to do to do modification to the same branch.

git clone https://github.com/cfm-mpc/pyscf.git
cd pyscf/
git branch -a

You should see remotes/origin/nao_back at the end of the list. Then,

git fetch origin
git checkout -b nao_back origin/nao_back

Then you will be in the new branch with nao. You can now do changes.

commit,
git commit -m "test commit" pyscf/nao/m_tddft_iter.py
 and push,
git push --set-upstream origin nao_back
