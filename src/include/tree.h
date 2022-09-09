#include <queue>
#include <stddef.h>

using namespace std;

// refer to: https://www.cnblogs.com/MrRS/p/9599532.html

#define MIN(x, y)      ((x) < (y) ? (x) : (y))
#define MAX(x, y)      ((x) > (y) ? (x) : (y))
#define ALIGN(x, c);      {for(int i = 0; i<x; ++i) cout << c;}
int intlen(int _x) {
  int len = 0;
  if (_x == 0)
    return 1;
  while (_x != 0) {
    _x /= 10;
    ++len;
  }
  return len;
}
string int2str(int _x) {
  string tmp, res;
  do {
    tmp.push_back((_x % 10) + '0');
    _x /= 10;
  } while (_x != 0);
  for (size_t index = tmp.size() - 1; index != -1; --index)
    res.push_back(tmp[index]);
  return res;
}

struct BiTNode {
  int value;
  BiTNode *parent;
  BiTNode *lchild;
  BiTNode *rchild;
  int pos;
  BiTNode(int _value = 0, int _pos = 0, BiTNode *_lchild = nullptr, BiTNode *_rchild = nullptr) :
          value(_value), lchild(_lchild), rchild(_rchild), pos(_pos) {}
  BiTNode(int value, BiTNode *parent, BiTNode *lchild, BiTNode *rchild)
		  : value(value), pos(0), parent(parent), lchild(lchild), rchild(rchild) {}
};

class binaryTree {
 public:
  BiTNode *src_root;
  BiTNode *res_root;

 private:
  struct returnData {
    int l_pos;
    int r_pos;
    returnData(int _l_most_r = INT_MAX, int _r_most_l = INT_MIN) : l_pos(_l_most_r), r_pos(_r_most_l) {}
  };

  void nodesPosAdj(BiTNode *_head, int _changes) {
    if (_head == nullptr)
      return;
    if (_head->lchild)
      _head->lchild->pos += _changes;
    if (_head->rchild)
      _head->rchild->pos += _changes;
    nodesPosAdj(_head->lchild, _changes);
    nodesPosAdj(_head->rchild, _changes);
  }

  returnData binaryPosAdj(BiTNode *_root, BiTNode *_head) {
    int changes = 0;
    returnData tmp, lchild_data, rchild_data;

    if (_head == nullptr)
      return tmp;
    lchild_data = binaryPosAdj(_root, _head->lchild);
    rchild_data = binaryPosAdj(_root, _head->rchild);
    while (_head->lchild != nullptr && lchild_data.r_pos >= _head->pos) {
      changes = lchild_data.r_pos - _head->pos + 2;
      _head->pos += changes;
      _head->rchild->pos += changes;
      nodesPosAdj(_head->rchild, changes);
      if (_head != _root)
        rchild_data = binaryPosAdj(_root, _head->rchild);
    }
    while (_head->rchild != nullptr && rchild_data.l_pos <= _head->pos) {
      changes = _head->pos - rchild_data.l_pos + 2;
      _head->rchild->pos += changes;
      nodesPosAdj(_head->rchild, changes);
      //if (_head != _root)
      rchild_data = binaryPosAdj(_root, _head->rchild);
    }
    tmp.l_pos = MIN(_head->pos, lchild_data.l_pos);
    tmp.r_pos = MAX(_head->pos, rchild_data.r_pos);
    return tmp;
  }

  void binaryTreePrint_bkp(BiTNode *_root) {
#define FILLSPACE(src, nums, c);  { for(size_t index = 0; index < nums; ++index) src.push_back(c); }
    queue<BiTNode *>src;
    BiTNode *tmp = nullptr;
    int level = 0;
    int pre_pos = 0, min_pos = 0, pre_len = 0;
    vector<int> nodes_per_level(2, 0);  // level nodes;
    string chars;

    if (_root == nullptr)
      return;
    min_pos = binaryPosAdj(_root, _root).l_pos;
    min_pos = abs(min_pos);
    _root->pos += min_pos;
    src.push(_root);
    nodes_per_level[level] = 1;
    while (!src.empty()) {
      tmp = src.front();
      src.pop();
      ALIGN(tmp->pos - pre_pos - pre_len, ' ');
      pre_pos = tmp->pos;
      pre_len = intlen(tmp->value);
      cout << tmp->value;
      if (tmp->lchild) {
        tmp->lchild->pos += min_pos;
        src.push(tmp->lchild);
        nodes_per_level[level + 1]++;
      }
      if (tmp->rchild) {
        tmp->rchild->pos += min_pos;
        src.push(tmp->rchild);
        nodes_per_level[level + 1]++;
      }

      if (--nodes_per_level[level] == 0) {
        cout << endl;
        nodes_per_level.push_back(0);
        ++level;
        pre_pos = 0;
        pre_len = 0;
      }
    }
  }

  void refreshNodesPos(int _min_pos, BiTNode *_head) {
    if (_head == nullptr)
      return;
    _head->pos += _min_pos;
    if (_head->lchild)
      refreshNodesPos(_min_pos, _head->lchild);
    if (_head->rchild)
      refreshNodesPos(_min_pos, _head->rchild);
  }
 public:
  binaryTree() : src_root(nullptr), res_root(nullptr) {}
  void generateFixedBinaryTree(void) {
    queue<BiTNode *> src;
    BiTNode *tmp = nullptr;
    int x = 0, lchild = 0, rchild = 0;

    cout << "please enter root value: ";
    cin >> x;
    src.push(src_root = new BiTNode(x));
    while (!src.empty()) {
      tmp = src.front();
      src.pop();
      cout << "please enter " << tmp->value << "'s lchild and rchild: ";
      cin >> lchild >> rchild;
      if (lchild != -1)
        src.push(tmp->lchild = new BiTNode(lchild));
      if (rchild != -1)
        src.push(tmp->rchild = new BiTNode(rchild));
    }
  }

  void generateFixedBinaryTree(vector<int> _nodes) {
    BiTNode *tmp = nullptr;
    queue<BiTNode *> x;
    int index = 0;
    int min_pos = 0;

    if (_nodes[0] == '#')
      return;
    x.push(src_root = new BiTNode(_nodes[index++], 0));
    min_pos = src_root->pos;
    while (!x.empty()) {
      tmp = x.front();
      x.pop();
      if (_nodes[index] != '#') {
        x.push(tmp->lchild = new BiTNode(_nodes[index], (tmp->pos - intlen(_nodes[index]) - 1)));
        min_pos = MIN(min_pos, (tmp->pos - intlen(_nodes[index]) - 1));
      }
      ++index;
      (index >= _nodes.size()) ? (index=_nodes.size()-1, _nodes[index]='#') : 0;
      if (_nodes[index] != '#')
        x.push(tmp->rchild = new BiTNode(_nodes[index], (tmp->pos + (intlen(tmp->value) + 1))));
      ++index;
      (index >= _nodes.size()) ? (index=_nodes.size()-1, _nodes[index]='#') : 0;
    }
  }

  void generateRandomBinaryTree(int _maxsize, int _maxvalue) {
    BiTNode *tmp = nullptr;
    queue<BiTNode * > src;
    int nullvalue = (_maxvalue >> 1);
    int value = 0;
    int maxsize = (rand() % _maxsize);
    int size = 0;

    src_root = new BiTNode(rand() % _maxvalue);
    if (src_root->value == nullvalue)
      return;
    ++size;
    src.push(src_root);
    while ((!src.empty()) && (size < maxsize)) {
      tmp = src.front();
      src.pop();
      if (tmp->value != '#') {
        value = rand() % _maxvalue;
        if (value != nullvalue) {//((value > (nullvalue - 10)) && (value > (nullvalue + 10))) {
          ++size;
          src.push(tmp->lchild = new BiTNode(value));
        }
        else
          src.push(tmp->lchild = new BiTNode('#'));
        value = rand() % _maxvalue;
        if (value != nullvalue) {//((value > (nullvalue - 10)) && (value > (nullvalue + 10))) {
          ++size;
          src.push(tmp->rchild = new BiTNode(value));
        }
        else
          src.push(tmp->rchild = new BiTNode('#'));
      }
    }
  }

  void generateRandomBinaryTree(vector<int> _nodes) {
    int flag = 1;  //0: lchild    1: rchild    2: lchild&&rchild
    queue<BiTNode *> x;
    BiTNode *tmp = nullptr;
    int index = 1;

    src_root = new BiTNode(_nodes[0]);
    x.push(src_root);
    while (index < _nodes.size() && !x.empty()) {
      tmp = x.front();
      x.pop();
      flag = (rand() % 3);  //flag = 0 / 1 / 2
      if (flag == 0) {
        tmp->lchild = new BiTNode(_nodes[index++]);
        x.push(tmp->lchild);
      }
      else if (flag == 1) {
        tmp->rchild = new BiTNode(_nodes[index++]);
        x.push(tmp->rchild);
      }
      else {
        tmp->lchild = new BiTNode(_nodes[index++]);
        tmp->rchild = new BiTNode(_nodes[index++]);
        x.push(tmp->lchild);
        x.push(tmp->rchild);
      }
    }
  }

  void binaryTreePrint(void) {
    queue<BiTNode *>src;
    BiTNode *tmp = nullptr;
    int level = 0;
    int pre_num_pos = 0, pre_line_pos = 0, min_pos = 0, pre_len = 0, fill_len = 0;
    vector<int> nodes_per_level(2, 0);  //level nodes;
    vector<string> chars(1);
    vector<string> nums(1);

    if (src_root == nullptr)
      return;
    min_pos = binaryPosAdj(src_root, src_root).l_pos;
    refreshNodesPos(abs(min_pos), src_root);
    src.push(src_root);
    nodes_per_level[level] = 1;
    while (!src.empty()) {
      tmp = src.front();
      src.pop();
      fill_len = tmp->pos - pre_num_pos - pre_len;
      chars[level] += *new string((tmp->pos - pre_line_pos - pre_len), ' ');
      chars[level] += pre_len > 1 ? *new string(pre_len - 1, ' ') : "";
      chars[level] += '|';
      if (tmp->lchild) {
        nums[level] += *new string(tmp->lchild->pos - (tmp->pos - fill_len), ' ');
        nums[level] += *new string(tmp->pos - tmp->lchild->pos, '-');
      }
      else
        nums[level] += *new string(fill_len, ' ');
      nums[level] += int2str(tmp->value);
      pre_num_pos = tmp->pos;
      pre_line_pos = tmp->pos;
      pre_len = intlen(tmp->value);
      if (tmp->rchild) {
        int len = intlen(tmp->rchild->value);
        len = tmp->rchild->pos - tmp->pos - (len > 1 ? len - 1 : 0);
        nums[level] += *new string(len, '-');
        pre_num_pos += len;
      }
      if (tmp->lchild) {
        src.push(tmp->lchild);
        nodes_per_level[level + 1]++;
      }
      if (tmp->rchild) {
        src.push(tmp->rchild);
        nodes_per_level[level + 1]++;
      }
      if (--nodes_per_level[level] == 0) {
        nums[level] += '\n';
        nums.push_back("");
        chars[level] += '\n';
        chars.push_back("");
        nodes_per_level.push_back(0);
        ++level;
        pre_num_pos = 0;
        pre_line_pos = 0;
        pre_len = 0;
      }
    }

    int index_nums = 0, index_chars = 0;
    for (size_t index = 0; index < nums.size() - 1; ++index) {
      cout << nums[index_nums++];
      cout << chars[++index_chars];
    }
  }
};
