import os
os.environ['CURL_CA_BUNDLE'] = ''
import logging
from re import template
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer

import config
import data_loader
from model import Casrel
from utils.common_utils import set_seed, set_logger, read_json, fine_grade_tokenize
from utils.train_utils import load_model_and_parallel, build_optimizer_and_scheduler, save_model
from utils.metric_utils import calculate_metric_relation, get_p_r_f
from tensorboardX import SummaryWriter

args = config.Args().get_parser()
set_seed(args.seed)
logger = logging.getLogger(__name__)

if args.use_tensorboard == "True":
    writer = SummaryWriter(log_dir='./tensorboard')
  
def get_spo(object_preds, subject_ids, length, example, id2tag):
  # object_preds:[batchsize, maxlen, num_labels, 2]
  num_label = object_preds.shape[2]
  num_subject = len(subject_ids)
  relations = []
  subjects = []
  objects = []
  # print(object_preds.shape, subject, length, example)
  for b in range(num_subject):
    tmp = object_preds[b, ...]
    subject_start, subject_end = subject_ids[b].cpu().numpy()
    subject = example[subject_start:subject_end+1]
    if subject not in subjects:
      subjects.append(subject)
    for label_id in range(num_label):
      start = tmp[:, label_id, :1]
      end = tmp[:, label_id, 1:]
      start = start.squeeze()[:length]
      end = end.squeeze()[:length]
      for i, st in enumerate(start):
        if st > 0.5:
          s = i
          for j in range(i, length):
            if end[j] > 0.5:
              e = j
              object = example[s:e+1]
              if object not in objects:
                objects.append(object)
              if (subject, id2tag[label_id], object) not in relations:
                relations.append((subject, id2tag[label_id], object))
              break
  # print(relations) 
  return relations, subjects, objects
  


def get_subject_ids(subject_preds, mask):
  lengths = torch.sum(mask, -1)
  starts = subject_preds[:, :, :1]
  ends = subject_preds[:, :, 1:]
  subject_ids = []
  for start, end, l in zip(starts, ends, lengths):
    tmp = []
    start = start.squeeze()[:l]
    end = end.squeeze()[:l]
    for i, st in enumerate(start):
      if st > 0.5:
        s = i
        for j in range(i, l):
          if end[j] > 0.5:
            e = j
            if (s,e) not in subject_ids:
              tmp.append([s,e])
            break

    subject_ids.append(tmp)
  return subject_ids

class BertForRe:
    def __init__(self, args, train_loader, dev_loader, test_loader, id2tag, tag2id, model, device):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.args = args
        self.id2tag = id2tag
        self.tag2id = tag2id
        self.model = model
        self.device = device
        if train_loader is not None:
          self.t_total = len(self.train_loader) * args.train_epochs
          self.optimizer, self.scheduler = build_optimizer_and_scheduler(args, model, self.t_total)

    def train(self):
        # Train
        global_step = 0
        self.model.zero_grad()
        eval_steps = 500 #每多少个step打印损失及进行验证
        best_f1 = 0.0
        for epoch in range(self.args.train_epochs):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for batch in batch_data[:-1]:
                    batch = batch.to(self.device)
                # batch_token_ids, attention_mask, token_type_ids, batch_subject_labels, batch_object_labels, batch_subject_ids
                loss = self.model(batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4], batch_data[5])

                # loss.backward(loss.clone().detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                logger.info('【train】 epoch:{} {}/{} loss:{:.4f}'.format(epoch, global_step, self.t_total, loss.item()))
                global_step += 1
                if self.args.use_tensorboard == "True":
                    writer.add_scalar('data/loss', loss.item(), global_step)
                if global_step % eval_steps == 0:
                    precision, recall, f1_score = self.dev()
                    logger.info('[eval] precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(precision, recall, f1_score))
                    if f1_score > best_f1:
                        save_model(self.args, self.model, model_name, global_step)
                        best_f1 = f1_score


    def dev(self):
      self.model.eval()
      spos = []
      true_spos = []
      subjects = []
      objects = []
      all_examples = []
      with torch.no_grad():
          for eval_step, dev_batch_data in enumerate(self.dev_loader):
              for dev_batch in dev_batch_data[:-1]:
                  dev_batch = dev_batch.to(self.device)
              
              seq_output, subject_preds = self.model.predict_subject(dev_batch_data[0], dev_batch_data[1],dev_batch_data[2])
              # 注意这里需要先获取subject，然后再来获取object和关系，和训练直接使用subject_ids不一样
              cur_batch_size = dev_batch_data[0].shape[0]
              dev_examples = dev_batch_data[-1]
              true_spos += [i[1] for i in dev_examples]
              all_examples += [i[0] for i in dev_examples]
              subject_labels = dev_batch_data[3].cpu().numpy()
              object_labels = dev_batch_data[4].cpu().numpy()
              subject_ids = get_subject_ids(subject_preds, dev_batch_data[1])

              example_lengths = torch.sum(dev_batch_data[1].cpu(), -1)
              
              for i in range(cur_batch_size):
                seq_output_tmp = seq_output[i, ...]
                subject_ids_tmp = subject_ids[i]
                length = example_lengths[i]
                example = dev_examples[i][0]
                if subject_ids_tmp:
                  seq_output_tmp = seq_output_tmp.unsqueeze(0).repeat(len(subject_ids_tmp), 1, 1)
                  subject_ids_tmp = torch.tensor(subject_ids_tmp, dtype=torch.long, device=device)
                  if len(seq_output_tmp.shape) == 2:
                    seq_output_tmp = seq_output_tmp.unsqueeze(0)
                  object_preds = model.predict_object([seq_output_tmp, subject_ids_tmp])
                  spo, subject, object = get_spo(object_preds, subject_ids_tmp, length, example, self.id2tag)
                  spos.append(spo)
                  subjects.append(subject)
                  objects.append(object)
                else:
                  spos.append([])
                  subjects.append([])
                  objects.append([])

          # for m,n, ex in zip(spos, true_spos, all_examples):
          #   print(ex)
          #   print(m, n)
          #   print('='*100)
          tp, fp, fn = calculate_metric_relation(spos, true_spos)
          p, r, f1 = get_p_r_f(tp, fp, fn)
          # print("========metric========")
          # print("precision:{} recall:{} f1:{}".format(p, r, f1))

          return p, r, f1
                
                

    def test(self, model_path):
        model = Casrel(self.args, self.tag2id)
        model, device = load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        spos = []
        true_spos = []
        subjects = []
        objects = []
        all_examples = []
        with torch.no_grad():
            for eval_step, dev_batch_data in enumerate(dev_loader):
                for dev_batch in dev_batch_data[:-1]:
                    dev_batch = dev_batch.to(device)
                
                seq_output, subject_preds = model.predict_subject(dev_batch_data[0], dev_batch_data[1],dev_batch_data[2])
                # 注意这里需要先获取subject，然后再来获取object和关系，和训练直接使用subject_ids不一样
                cur_batch_size = dev_batch_data[0].shape[0]
                dev_examples = dev_batch_data[-1]
                true_spos += [i[1] for i in dev_examples]
                all_examples += [i[0] for i in dev_examples]
                subject_labels = dev_batch_data[3].cpu().numpy()
                object_labels = dev_batch_data[4].cpu().numpy()
                subject_ids = get_subject_ids(subject_preds, dev_batch_data[1])

                example_lengths = torch.sum(dev_batch_data[1].cpu(), -1)
                
                for i in range(cur_batch_size):
                  seq_output_tmp = seq_output[i, ...]
                  subject_ids_tmp = subject_ids[i]
                  length = example_lengths[i]
                  example = dev_examples[i][0]
                  if subject_ids_tmp:
                    seq_output_tmp = seq_output_tmp.unsqueeze(0).repeat(len(subject_ids_tmp), 1, 1)
                    subject_ids_tmp = torch.tensor(subject_ids_tmp, dtype=torch.long, device=device)
                    if len(seq_output_tmp.shape) == 2:
                      seq_output_tmp = seq_output_tmp.unsqueeze(0)
                    object_preds = model.predict_object([seq_output_tmp, subject_ids_tmp])
                    spo, subject, object = get_spo(object_preds, subject_ids_tmp, length, example, self.id2tag)
                    spos.append(spo)
                    subjects.append(subject)
                    objects.append(object)
                  else:
                    spos.append([])
                    subjects.append([])
                    objects.append([])



            for i, (m,n, ex) in enumerate(zip(spos, true_spos, all_examples)):
              if i <= 10:
                print(ex)
                print(m, n)
                print('='*100)
            tp, fp, fn = calculate_metric_relation(spos, true_spos)
            p, r, f1 = get_p_r_f(tp, fp, fn)
            print("========metric========")
            print("precision:{} recall:{} f1:{}".format(p, r, f1))

            return p, r, f1

    def predict(self, raw_text, model, tokenizer):
        model.eval()
        with torch.no_grad():
            tokens = [i for i in raw_text]
            if len(tokens) > self.args.max_seq_len:
              tokens = tokens[:self.args.max_seq_len]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_masks = [1] * len(token_ids)
            token_type_ids = [0] * len(token_ids)
            if len(token_ids) < self.args.max_seq_len:
              token_ids = token_ids + [0] * (self.args.max_seq_len - len(tokens))
              attention_masks = attention_masks + [0] * (self.args.max_seq_len - len(tokens))
              token_type_ids = token_type_ids + [0] * (self.args.max_seq_len - len(tokens))
            assert len(token_ids) == self.args.max_seq_len
            assert len(attention_masks) == self.args.max_seq_len
            assert len(token_type_ids) == self.args.max_seq_len
            token_ids = torch.from_numpy(np.array(token_ids)).unsqueeze(0).to(self.device)
            attention_masks = torch.from_numpy(np.array(attention_masks, dtype=np.uint8)).unsqueeze(0).to(self.device)
            token_type_ids = torch.from_numpy(np.array(token_type_ids)).unsqueeze(0).to(self.device)
            seq_output, subject_preds = model.predict_subject(token_ids, attention_masks, token_type_ids)
            subject_ids = get_subject_ids(subject_preds, attention_masks)

            cur_batch_size = seq_output.shape[0]
            spos = []
            subjects = []
            objects = []
            for i in range(cur_batch_size):
                seq_output_tmp = seq_output[i, ...]
                subject_ids_tmp = subject_ids[i]
                length = len(tokens)
                example = raw_text
                if any(subject_ids_tmp):
                  seq_output_tmp = seq_output_tmp.unsqueeze(0).repeat(len(subject_ids_tmp), 1, 1)

                  subject_ids_tmp = torch.tensor(subject_ids_tmp, dtype=torch.long, device=device)
                  if len(seq_output_tmp.shape) == 2:
                    seq_output_tmp = seq_output_tmp.unsqueeze(0)
                  object_preds = model.predict_object([seq_output_tmp, subject_ids_tmp])

                  spo, subject, object = get_spo(object_preds, subject_ids_tmp, length, example, self.id2tag)

                  subjects.append(subject)
                  objects.append(object)
                  spos.append(spo)
            print("文  本：", raw_text)
            print('主  体：', subjects)
            print('客  体：', objects)
            print('三元组：', spos)
            print("="*100)

if __name__ == '__main__':
    data_name = 'ske'
    model_name = 'bert'
    set_logger(os.path.join(args.log_dir, '{}.log'.format(model_name)))
    if data_name == "ske":
        args.data_dir = './data/ske'
        data_path = os.path.join(args.data_dir, 'raw_data')
        label_list = read_json(os.path.join(args.data_dir, 'mid_data'), 'predicates')
        tag2id = {}
        id2tag = {}
        # 建立 predicate 与 索引之间的双向映射，方便查询
        for k,v in enumerate(label_list):
            tag2id[v] = k
            id2tag[k] = v
        
        logger.info(args)
        max_seq_len = args.max_seq_len
        tokenizer = BertTokenizer.from_pretrained('model_hub/chinese-bert-wwm-ext/vocab.txt')

        model = Casrel(args, tag2id)
        model, device = load_model_and_parallel(model, args.gpu_ids)

        collate = data_loader.Collate(max_len=max_seq_len, tag2id=tag2id, device=device, tokenizer=tokenizer)


        train_dataset = data_loader.MyDataset(file_path=os.path.join(data_path, 'train_data.json'), 
                    tokenizer=tokenizer, 
                    max_len=max_seq_len)

        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate.collate_fn) 
        dev_dataset = data_loader.MyDataset(file_path=os.path.join(data_path, 'dev_data.json'), 
                    tokenizer=tokenizer, 
                    max_len=max_seq_len)

        # 快速验证，使用部分的验证集数据
        dev_dataset = dev_dataset[:args.use_dev_num]
        dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate.collate_fn) 


        bertForNer = BertForRe(args, train_loader, dev_loader, dev_loader, id2tag, tag2id, model, device)
        bertForNer.train()

        model_path = './checkpoints/bert/model.pt'.format(model_name)
        bertForNer.test(model_path)

        # texts = [
        #     '皮肤鳞状细胞癌@电离辐射、砷或者焦油、人乳头瘤病毒、免疫抑制、既往皮肤癌以及遗传皮肤病可能会增加鳞状细胞癌的患病风险',
        #     '慢性心房颤动@明显可触及且规律的脉搏并不能排除心房颤动。 慢性心房颤动@体检也可能发现与心房颤动潜在病因相关的表现，如心力衰竭的表现、卒中或内分泌疾病，如甲状腺功能亢进',
        #     '铅中毒@### 异食癖 儿童出现喜欢吃非营养物质的疾病。',
        #     '急性髓性白血病@坏疽性脓皮病的特点是存在腿部溃疡或不太常见的手部溃疡。急性髓性白血病@其出现是免疫功能障碍所导致的结果之一，并且可能与 AML 有关',
        #     'IRS Ⅲ的结果显示，膀胱和前列腺部横纹肌肉瘤患者在接受加强化疗和早期放疗后，其成活率仅次于头颈部横纹肌肉瘤患者。',
        #     '（2）罗马Ⅱ标准（1999年制定）： 小儿CVS诊断标准：①3个或3个周期以上剧烈的恶心、顽固性呕吐，持续数小时到数日，间隙期持续数日到数月；②排除代谢性、胃肠道及中枢神经系统器质性疾病',
        # ]
        
        # texts = [
        # '小婴儿有时仅表现为激惹、睡眠规律紊乱和喂养困难。 根据病理组织学和病情发展，肝功能衰竭可以分为急性肝衰竭（acute hepatic failure，AHF）、亚急性肝衰竭（subacute hepatic failure，SHF）、慢加急性肝衰竭（acute-on-chronic hepatic failure，ACHF）和慢性肝衰竭（chronic hepatic failure，CHF）。',
        # '人蛔虫病是世界上流行最广的人类蠕虫病,据世界卫生组织（WHO)估计全球有13亿患者，儿童，特别是学龄前儿童感染率高。轻者无任何症状，大量蛔虫感染可引起食欲缺乏或多食易饥，异食癖;常腹痛，位于脐周， 喜按揉，不剧烈;部分患者烦躁易惊或萎靡、磨牙;虫体的异种蛋白可引起荨麻疹、哮喘等过敏症状。',
        # 'B族链球菌感染@脓毒症 * 一线疗法：青霉素或氨苄西林。B族链球菌感染@ * 青霉素过敏患者：二代或三代头孢菌素（可能适用，具体取决于过敏反应类型）或者万古霉素。',
        # '消化性溃疡病@如果患者有报警症状，则需要在治疗前行内镜检查。',
        # '在高渗性脱水，水从细胞内转移至细胞外使细胞内外的渗透压达到平衡，其结果是细胞内容量降低。由于细胞内缺水，患儿常有剧烈口渴、高热、烦躁不安、肌张力增高等表现，甚至发生惊厥。'
        # ]

        texts = [
            '【临床表现及诊断】 对MAS临床诊断主要有以下方面： （一）宫内窘迫史 有宫内窘迫或产时窒息者，可以在出生后1、5、10分钟进行Apgar评分，低于3分，为严重窒息可能。 （三）临床出现呼吸困难症状 一般表现为进行性呼吸困难，有肋间凹陷征',
            '慢性胰腺炎@原发性胰管假说认为胰腺炎损伤首先发生在胰管，属于一种原发性自身免疫性反应或炎性反应，然而前哨急性胰腺炎假说 认为损伤首先发生在腺泡细胞，触发炎症细胞的扣留和细胞因子的分泌。',
            'MMA临床表现无特异性，易于漏诊或误诊，最常见的症状是反复呕吐、嗜睡、惊厥、运动障碍、智力及肌张力低下。 甲硝唑10~20mg/(kg• d)或新霉素50mg/(kg • d),可减少肠道细菌产生的丙酸,但长期应用可引起肠道菌群紊乱，应慎用。',
            '弥漫性大B细胞淋巴瘤（diffuse large B-cell lymphoma）是一种好发于大龄青少年的疾病，其发病率在整个童年期稳步上升，并在15～19岁年龄组内作为主导地位的组织学亚型达到高峰',
            '自由基以及抗氧化剂缺乏在慢性胰腺炎的形成和发展中都起了很重要的作用。严重时可出现脂肪泻，患儿粪便量显著增多，粪酸臭或恶臭',
            '皮肤鳞状细胞癌@电离辐射、砷或者焦油、人乳头瘤病毒、免疫抑制、既往皮肤癌以及遗传皮肤病可能会增加鳞状细胞癌的患病风险',
            '慢性心房颤动@明显可触及且规律的脉搏并不能排除心房颤动。 慢性心房颤动@体检也可能发现与心房颤动潜在病因相关的表现，如心力衰竭的表现、卒中或内分泌疾病，如甲状腺功能亢进',
            '按病程发展及主要临床表现，可分为急性、慢性及晚期血吸虫病。（一）急性血吸虫病 多见于夏秋季，以小儿及青壮年为多。',
            '急性髓性白血病@坏疽性脓皮病的特点是存在腿部溃疡或不太常见的手部溃疡。急性髓性白血病@其出现是免疫功能障碍所导致的结果之一，并且可能与 AML 有关',
            '发热伴肝、脾肿大，可见于传染性单核细胞增多症、疟疾、黑热病、急性血吸虫病、结缔组织疾病、白血病及恶性淋巴瘤等。',
            '（2）罗马Ⅱ标准（1999年制定）： 小儿CVS诊断标准：①3个或3个周期以上剧烈的恶心、顽固性呕吐，持续数小时到数日，间隙期持续数日到数月；②排除代谢性、胃肠道及中枢神经系统器质性疾病',
        ]



        model = Casrel(args, tag2id)
        model, device = load_model_and_parallel(model, args.gpu_ids, model_path)
        for text in texts:
          bertForNer.predict(text, model, tokenizer)

