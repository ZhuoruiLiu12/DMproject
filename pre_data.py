import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_label4offline_train(row):
    """
    todo:为offline_train创建label 1-表示在15天内使用了优惠券，0-表示没有使用，-1-表明没有领优惠劵
    @param row:
    @return:
    """
    # if pd.isnull(row['Date']) and (not pd.isnull(row['Coupon_id'])):
    if pd.isnull(row['Date']):
        if not pd.isnull(row['Coupon_id']):
            # 如果Date=null & Coupon_id != null，该记录表示领取优惠券但没有使用，即负样本
            return 0
    else:
        # Date != null
        if pd.isnull(row['Coupon_id']):
            # Coupon_id = null，则表示普通消费日期
            return -1
        else:
            # 如果Date!=null & Coupon_id != null，则表示用优惠券消费日期，即正样本
            return 1


def pre_process_data(df):
    """
    todo: 预处理数据,包括构建label、去除无用数据
    @return:
    """
    # *添加label
    df['label'] = df.apply(create_label4offline_train, axis=1)
    print(df['label'].value_counts())
    # *去除没有接收领优惠卷的用户,因为不是我们这次分析的样本，需要从样本空间删除
    df = df[df['label'] != -1]
    print(df['label'].value_counts())
    # *保存数据
    df.to_csv('processed_offline_train.csv', index=False)


def user_count(data):
    """
    @todo 区分高活跃度用户-3，中度活跃用户-2以及低活跃度用户-1，以及流失用户-0
    @param data:
    @return:
    """
    if data > 10:
        return 3
    elif data > 5:
        return 2
    elif data > 1:
        return 1
    else:
        return 0


def user_id(df):
    """
    todo: 分析user id属性
    @return:
    """
    print("total number is %d" % len(df.User_id))
    print("unique user id is %d" % len(set(df.User_id)))
    df['sum'] = 1  # 给每一条记录付给一个初始值1,记录出现的次数
    user_id_count = df.groupby(['User_id'], as_index=False)['sum'].agg({'count': np.sum})  # 统计各个用户出现的次数
    user_id_count.sort_values('count', ascending=False, inplace=True)
    print(user_id_count.head(20))
    # 可视化
    user_id_count['user_range'] = user_id_count['count'].map(user_count)
    f, ax = plt.subplots(1, 2, figsize=(17.5, 8))
    user_id_count['user_range'].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax[0], explode=[0.01, 0, 0, 0],
                                                    startangle=90)
    ax[0].set_title('user_range_ratio')
    ax[0].set_ylabel('')
    # user_id_count['user_range'].value_counts().plot(kind='bar',ax=ax[1])
    sns.countplot(x='user_range', data=user_id_count, ax=ax[1])
    ax[1].set_title('user range distribution')
    plt.savefig("用户活跃度.png")
    plt.show()


def mer_count(data):
    if data > 1000:
        return 3
    elif data > 100:
        return 2
    elif data > 20:
        return 1
    else:
        return 0


def merchant_id(df):
    """
    todo: 分析merchant id属性
    @param df:
    @return:
    """
    print("total number is %d" % len(df.Merchant_id))
    print("unique user id is %d" % len(set(df.Merchant_id)))
    print("total number is %d" % len(df.Merchant_id))
    print("unique user id is %d" % len(set(df.Merchant_id)))
    df['sum_Merchant_id'] = 1  # 给每一条记录付给一个初始值1,记录出现的次数
    merchant_count = df.groupby(['Merchant_id'], as_index=False)['sum_Merchant_id'].agg(
        {'count': np.sum})  # 统计各个用户出现的次数
    merchant_count.sort_values('count', ascending=False, inplace=True)
    print(len(merchant_count))
    print(merchant_count.head(20))
    # visualization
    merchant_count['mer_range'] = merchant_count['count'].map(mer_count)  # 对商家也进行编码，3代表发放了很多的优惠券，2,1,0一次抵减
    f, ax = plt.subplots(1, 2, figsize=(17.5, 8))
    merchant_count['mer_range'].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax[0], explode=[0.01, 0, 0, 0],
                                                    startangle=90)
    ax[0].set_title('mer_range')
    ax[0].set_ylabel('')

    sns.countplot(x='mer_range', data=merchant_count, ax=ax[1])
    ax[1].set_title('merchant range distribution')
    plt.savefig("商家分布.png")
    plt.show()


def cou_count(data):
    if data > 1000:
        return 3
    elif data > 100:
        return 2
    elif data > 10:
        return 1
    else:
        return 0


def coupon_id(df):
    """
    @todo 分析coupon id属性
    @param df:
    @return:
    """
    # 观察有发放优惠券的商家行为
    cou_id_count = df.groupby(['Coupon_id'], as_index=False)['sum'].agg({'count': np.sum})
    cou_id_count.sort_values(["count"], ascending=False).head(10)
    # visualization
    cou_id_count['Cou_range'] = cou_id_count['count'].map(cou_count)
    # 绘图
    f, ax = plt.subplots(1, 2, figsize=(17.5, 8))
    cou_id_count['Cou_range'].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax[0], explode=[0.01, 0, 0, 0],
                                                  startangle=90)
    ax[0].set_title('Cou_range')
    ax[0].set_ylabel('')

    sns.countplot(x='Cou_range', data=cou_id_count, ax=ax[1])
    ax[1].set_title('Cou_range distribution')
    plt.savefig("优惠券分布.png")
    plt.show()


def convertRate(row):
    """Convert discount to rate"""
    if pd.isnull(row):
        return 1.0
    elif ':' in row:
        rows = row.split(':')
        return np.round(1.0 - float(rows[1]) / float(rows[0]), 2)
    else:
        return float(row)


def convertDistance(row):
    """Convert Distance to rate"""
    if pd.isnull(row):
        row = -1
    return int(row)


def date(df):
    """
    @todo 分析Date_received和Date属性
    @param df:
    @return:
    """
    # couponbydate = df[['Date_received', 'Date']].groupby(['Date_received'],as_index=False).count()
    couponbydate = df[df['Date_received'] != 'null'][['Date_received', 'Date']].groupby(
        ['Date_received'],
        as_index=False).count()
    couponbydate.columns = ['Date_received', 'count']
    # df_temp = df[~df['Date'].isna()]
    # buybydate = df_temp[['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
    buybydate = \
        df[(df['Date'] != 'null') & (df['Date_received'] != 'null')][
            ['Date_received', 'Date']].groupby(
            ['Date_received'], as_index=False).count()
    buybydate.columns = ['Date_received', 'count']
    date_buy = df['Date'].unique()  # 购物的日期
    date_buy = sorted(date_buy[~np.isnan(date_buy)])  # 按照购物的日期进行排，排出null值

    date_received = df['Date_received'].unique()  # 接收到优惠券的时间
    date_received = sorted(date_received[~np.isnan(date_received)])  # 按照接收优惠券的时间排序

    sns.set_style('ticks')
    sns.set_context("notebook", font_scale=1.4)
    plt.figure(figsize=(12, 8))
    date_received = [int(x) for x in date_received]
    date_received_dt = pd.to_datetime(date_received, format='%Y%m%d')  # 转换为datetime格式
    plt.subplot(211)
    plt.bar(date_received_dt, couponbydate['count'], label='number of coupon received',
            color='#a675a1')  # 绘制接收优惠券的日期对应的bar图
    plt.bar(date_received_dt, buybydate['count'], label='number of coupon used', color='#75a1a6')  # 绘制使用优惠券对应的bar图
    plt.yscale('log')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig("优惠券使用图.png")
    plt.show()

    plt.subplot(212)
    plt.bar(date_received_dt, buybydate['count'] / couponbydate['count'], color='#62a5de')  # 绘制优惠券使用的比例图
    plt.ylabel('Ratio(coupon used/coupon received)')
    plt.tight_layout()
    plt.savefig("优惠券使用比例图.png")
    plt.show()


def attr_corr(df):
    """
    @todo 分析属性之间相关关系并作图
    @param df:
    @return:
    """
    df_corr = df.corr()
    print(df_corr)
    plt.subplots(figsize=(12, 9))
    sns.heatmap(df_corr, square=True, vmax=1, vmin=-1, center=0)
    plt.savefig("属性相关性heatmap.png")
    plt.show()


def analysis_data(df):
    """
    todo: 分析数据质量、分布
    """
    # *分析重复数据
    print(df_pro.duplicated().value_counts())
    # *分析缺失值
    print(df_pro.isnull().sum())
    # *分析User_id
    # user_id(df)
    # *分析商家
    # merchant_id(df)
    # *分析优惠券
    # coupon_id(df)
    # *分析优惠力度
    df['Discount_rate'] = df['Discount_rate'].apply(convertRate)
    # (df['Discount_rate'].value_counts() / len(df)).plot(kind='bar', title="discount_rate")
    # plt.savefig("优惠力度统计.png")
    # plt.show()
    # *分析距离
    df['Distance'] = df['Distance'].apply(convertDistance)
    # df['Distance'].unique()
    # (df['Distance'].value_counts() / len(df)).plot(kind='bar',title='distance from merchant')
    # plt.savefig("商家距离统计.png")
    # plt.show()
    # *分析Date_received和Date属性
    # date(df)
    # *分析属性之间相关关系
    attr_corr(df)



if __name__ == '__main__':
    # offline_train_data = pd.read_csv('offline_train.csv')
    # online_train_data = pd.read_csv('datasets/online_train.csv/online_train.csv')
    # offline_test_data = pd.read_csv('datasets/offline_test.csv/offline_test.csv')
    # pre_process_data(offline_train_data)
    df_pro = pd.read_csv('datasets/offline_train.csv/processed_offline_train.csv')
    analysis_data(df_pro)